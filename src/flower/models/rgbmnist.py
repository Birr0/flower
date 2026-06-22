import math 

import lightning as L
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper, gradient
from timm.models.layers import trunc_normal_
from torch import Tensor
from huggingface_hub import PyTorchModelHubMixin

from flower.models.modules import get_conditional_len, BaseModel, AdaLN, TimestepEmbedder, \
    WrappedModel, ConditionEmbedder, ConditionalPrior

class VelocityField(nn.Module, PyTorchModelHubMixin):
    def __init__(self, code_dim, hidden_dim, conditional_dim, n_hidden=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = 1

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.input_proj = nn.Linear(code_dim, hidden_dim)

        self.act = nn.SiLU()
        self.ada_lns = nn.ModuleList(
            [AdaLN(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        self.out_proj = nn.Linear(hidden_dim, code_dim) 

        self.cond_embed = ConditionEmbedder(
            conditional_dim, hidden_dim
        )
        self.null_y = nn.Embedding(
            num_embeddings=1,
            embedding_dim=conditional_dim,
        )

        self.conditional_prior = ConditionalPrior(
            cond_dim=conditional_dim,
            hidden_dim=hidden_dim,
            code_dim=code_dim
        )
        # parameter for y embeddings

    def forward(self, x_t: Tensor, t: Tensor, y: Tensor):
        t_embed = self.t_embedder(t).flatten(start_dim=1)
        y_embed = self.cond_embed(y)

        x = self.input_proj(x_t)
        c = t_embed + y_embed

        for adaln, lin in zip(self.ada_lns, self.linears):
            identity = x
            modulated, gate = adaln(x, c) # replace with c here.
            x = self.act(lin(modulated))
            x = identity + gate*x
        return self.out_proj(x)

class LightningFlowMatching(L.LightningModule):
    def __init__(
        self,
        base_model: BaseModel,
        lr,
        batch_size,
        code_dim,
        hidden_dim,
        catalog,
        n_steps=20,
        ckpt_path: str = None,
        method="midpoint",
        base_model_ckpt_path=None,
        beta_start_step=0,
        beta_warmup_steps=10000,
        max_beta=1.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.cond_dim = get_conditional_len(catalog)
        self.lr = lr

        self.beta_start_step = beta_start_step
        self.beta_warmup_steps = beta_warmup_steps
        self.max_beta = max_beta 

        # --- Models --- #
        self.vf = VelocityField(code_dim, hidden_dim, self.cond_dim)
        self.vf.apply(self._init_weights)
        self.base_model = base_model

        if base_model_ckpt_path:
            base_model_state_dict = torch.load(base_model_ckpt_path)[
                "state_dict"
            ] 
            base_model_state_dict = {
                k.replace("vae.", "", 1): v for k, v in base_model_state_dict.items()
            }
            # drop any encoder 0 blocks
            # or don't enforce strict here.
            
            self.base_model.load_state_dict(base_model_state_dict, strict=False) # 
            print("✅ Base model weights loaded.")

        # 2. Freeze the base model
        self.base_model.eval() # Set to evaluation mode
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("❄️ Base model frozen.")

        # --- Load Checkpoints --- #
        if ckpt_path:
            self.vf_state_dict = torch.load(ckpt_path)[
                "state_dict"
            ]  # map_location="cpu"
            self.load_state_dict(self.vf_state_dict, strict=False)
            print("✅ Loaded state dict from checkpoint.")
            self.wrapped_vf = WrappedModel(self.vf)
            # ODE solver hparams
            self.n_steps = n_steps
            self.solver = ODESolver(velocity_model=self.wrapped_vf)
            self.wrapped_vf = WrappedModel(self.vf)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.method = method
        self.step_size = 1./n_steps

    @property
    def T(self):
        return torch.tensor([1., 0.], device=self.device)
        #torch.linspace(1, 0, self.n_steps, device=self.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        params = list(self.vf.parameters())

        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )
    
    def get_beta(self):
        if self.global_step < self.beta_start_step:
            return 0.0
        s = self.global_step - self.beta_start_step
        if s < self.beta_warmup_steps:
            return self.max_beta * (s / self.beta_warmup_steps)
        return self.max_beta

    def base_step(self, batch, partition):
        X = batch["X"]
        y = batch["y"]

        x_1, _, _ = self.base_model.encode(X)
        batch_size = x_1.shape[0]

        mu_model, log_var = self.vf.conditional_prior(
            y
        )
        eps = torch.randn_like(x_1)
        x_0_cond = mu_model + torch.exp(0.5 * log_var) * eps

        x_0_uncond = torch.randn_like(x_1)

        x_1 = torch.cat([x_1, x_1], dim=0)
        x_0 = torch.cat([x_0_cond, x_0_uncond], dim=0)
        
        null_idx = torch.zeros(batch_size, dtype=torch.long, device=y.device)
        y_null = self.vf.null_y(null_idx) 
        y = torch.cat([y, y_null], dim=0)

        t = torch.rand(batch_size, device=x_1.device).unsqueeze(-1)
        t = torch.cat([t, t], dim=0)
        x_t = t*x_1 + (1 - t)*x_0

        v_t = self.vf(x_t=x_t, y=y, t=t)
        v_tgt = x_1 - x_0

        cfm_loss = torch.pow(
            v_t - v_tgt, 2
        ).mean()

        kl_loss = (0.5 * torch.sum(torch.exp(log_var) + mu_model**2 - 1 - log_var, dim=-1)).mean()
        beta = self.get_beta()
        loss = cfm_loss + beta*kl_loss

        self.log(f"{partition}_loss", loss)
        self.log(f"{partition}_cfm_loss", cfm_loss)
        self.log(f"{partition}_kl_loss", kl_loss)
        self.log(f"{partition}_beta", beta)
        return loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch, _batch_idx):
        return self.base_step(batch, "test")

    def predict_step(self, X, y, embed_opt=["cond"]):
        self.eval()
        with torch.no_grad():
            output = {}
            code, _, _ = self.base_model.encode(X)
            if "orig" in embed_opt:
                output["orig"] = code
            
            # could reduce this to a single forward pass.
            if "cond" in embed_opt:
                output["cond"] = self.solver.sample(
                    x_init=code,
                    step_size=self.step_size,
                    y=y,
                    cfg_scale=1.0,
                    time_grid=self.T,
                    method=self.method,
                )

            if "uncond" in embed_opt:
                output["uncond"] = self.solver.sample(
                    x_init=code,
                    step_size=self.step_size,
                    y=y,
                    cfg_scale=0.0,
                    time_grid=self.T,
                    method=self.method,
                )
        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.layers = nn.Sequential(
            self.encoder_block1,
            self.encoder_block2,
            self.encoder_block3,
            self.encoder_block4,
        )

    def forward(self, x: Tensor):
        for block in self.layers:
            for layer in block:
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_output_dim = 512

        self.block1 = nn.Sequential(
            nn.Upsample(size=3, mode="nearest"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Upsample(size=7, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Upsample(size=14, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Upsample(size=28, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

        self.layers = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

    def forward(self, z: Tensor):
        x = z.view(z.size(0), 512, 1, 1)
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x


class VAE(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder_output_dim = 512

        self.project_to_z_dist = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
        )

        # before decoding.
        self.projection_up = nn.Sequential(
            nn.Linear(self.hidden_dim, self.encoder_output_dim)
        )

    def reparametrize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(logvar * 0.5)

    def encode(self, x):
        for block in self.encoder.layers:
            for layer in block:
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)

        x = x.view(x.size(0), -1)

        x = self.project_to_z_dist(x)

        # sample from the latent network
        mu, log_var = x.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -30.0, 20.0)

        z = self.reparametrize(mu, log_var).view(mu.size(0), -1)
        return z, mu, log_var

    def forward(self, x):
        for block in self.encoder.layers:
            for layer in block:
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)

        x = x.view(x.size(0), -1)

        x = self.project_to_z_dist(x)


        # sample from the latent network
        mu, log_var = x.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -30.0, 20.0)

        z = self.reparametrize(mu, log_var).view(mu.size(0), -1)

        z_ = self.projection_up(z)  # 64 -> 512
        x = z_.view(z_.size(0), self.encoder_output_dim, 1, 1)
        # decode
        for block in self.decoder.layers:
            x = block(x)
        x = x.view(x.size(0), 3, 28, 28)
        return {"z": z, "recon": x, "mu": mu, "log_var": log_var}

class LightningVAE(L.LightningModule):
    def __init__(self, vae, lr, batch_size, beta, vae_ckpt_path=None, ckpt_path=None):
        # ckpt_path is for create_lightning_loader
        # it not as clean as it could be. Focus on single or double ckpt_path arg.
        super().__init__()
        self.vae = vae  # torch.compile(vae)
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.mse = F.mse_loss
        self.ckpt_path = ckpt_path

        self.alpha = torch.tensor(100.0)  # recon loss weight

        if vae_ckpt_path:
            state_dict = torch.load(
                vae_ckpt_path
            )["state_dict"]
            state_dict = {k.replace("vae.", "", 1): v for k, v in state_dict.items()}
            self.vae.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.vae.parameters(), lr=self.lr)

    def base_step(self, batch, partition):
        output = self.vae(batch["X"])
        # z = self.vae.reparametrize(output["mu"], output["log_var"])
        recon_loss = self.alpha * self.mse(
            output["recon"], batch["X"], reduction="mean"
        )

        kl_loss = torch.sum(
            -0.5
            * (1 + output["log_var"] - output["log_var"].exp() - output["mu"].pow(2)),
            axis=1,
        ).mean()

        loss = recon_loss + self.beta * kl_loss

        self.log(f"{partition}_loss", loss.mean(), sync_dist=True)
        self.log(f"{partition}_kl_loss", kl_loss, sync_dist=True)
        self.log(f"{partition}_recon_loss", recon_loss, sync_dist=True)
        return loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch):
        return self.base_step(batch, "test")

    def predict_step(self, batch):
        output = self.vae(batch["X"])

        return {
            "X": batch["X"],
            "recon": output["recon"],
            "z": output["z"].flatten(start_dim=1),
            "catalog": batch["catalog"],
        }

