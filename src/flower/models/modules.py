import math
from typing import Protocol

import lightning as L
import torch
import torch.nn as nn
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from huggingface_hub import PyTorchModelHubMixin
from timm.layers import trunc_normal_
from torch import Tensor


class BaseModel(Protocol):
    latent_dim: int

    def encode(self, X: torch.Tensor) -> torch.Tensor: ...


def get_conditional_len(y_catalog: dict) -> int:
    """
    Get the context length for the flow model given dropped variables
    and the y_catalog.
    """
    total_size = sum(y_catalog["variables"][v]["size"] for v in y_catalog["variables"])
    drop_size = sum(
        y_catalog["variables"][v]["size"] for v in y_catalog["drop_variables"]
    )
    return int(total_size - drop_size)


def get_no_of_continuous_variables(y_catalog: dict) -> int:
    """
    Get the context length for the flow model given dropped variables
    and the y_catalog.
    """
    continuous_size = sum(
        y_catalog["variables"][v].get("continuous", 0) for v in y_catalog["variables"]
    )
    drop_size = sum(
        y_catalog["variables"][v].get("continuous", 0)
        for v in y_catalog["drop_variables"]
    )
    return int(continuous_size - drop_size)


class WrappedModel(nn.Module):
    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(self, x, t, **model_extras):
        cfg_scale = model_extras.get("cfg_scale", 1.0)
        y = model_extras.get("y")
        batch_size = x.shape[0]

        # Ensure t is the right shape for the concatenated batch
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        # If no guidance, just run conditional
        if cfg_scale == 1.0:
            return self.velocity_model(x_t=x, t=t, y=y)

        # 1. Create Null y for inference
        null_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        y_null = self.velocity_model.null_y(null_idx)

        # 2. Batch doubling
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)
        y_double = torch.cat([y, y_null], dim=0)

        # 3. Predict velocities
        v_double = self.velocity_model(x_t=x_double, t=t_double, y=y_double)

        # 4. Chunk and Guide
        v_cond, v_uncond = torch.chunk(v_double, chunks=2, dim=0)

        # Apply CFG formula
        return v_uncond + cfg_scale * (v_cond - v_uncond)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(
            t.dtype
        )
        return self.mlp(t_freq)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # We predict scale (gamma) and shift (beta)
        self.ada_lin = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim, bias=True),
        )

    def forward(self, x, cond_emb):
        # Generate params from conditioning
        gamma, beta, gate = self.ada_lin(cond_emb).chunk(3, dim=-1)

        # Apply normalization first, then scale and shift
        x = self.norm(x)
        x = (
            x * (1 + gamma) + beta
        )  # (1 + gamma) helps initialization stay near identity
        return self.linear(x), gate  # return gating mechanism


class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout \
    for classifier-free guidance.
    """

    def __init__(self, cond_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Linear(cond_dim, hidden_size)

    def forward(self, y):
        return self.embedding(y)


class ConditionalPrior(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int, code_dim: int):
        super().__init__()
        # self.code_dim = code_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * code_dim),
        )
        self._init_weights()

    def _init_weights(self):
        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, y: Tensor) -> Tensor:
        # return mu = 0, logvar=0
        mu, logvar = self.net(y).chunk(2, dim=-1)
        return mu, logvar


class VelocityField(nn.Module, PyTorchModelHubMixin):
    def __init__(self, code_dim, hidden_dim, conditional_dim, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = 1

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.input_proj = nn.Linear(code_dim, hidden_dim)

        self.act = nn.SiLU()
        self.ada_lns = nn.ModuleList(
            [AdaLN(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, code_dim)

        self.cond_embed = ConditionEmbedder(conditional_dim, hidden_dim)
        self.null_y = nn.Embedding(
            num_embeddings=1,
            embedding_dim=conditional_dim,
        )

        self.conditional_prior = ConditionalPrior(
            cond_dim=conditional_dim, hidden_dim=hidden_dim, code_dim=code_dim
        )
        # parameter for y embeddings

    def forward(self, x_t: Tensor, t: Tensor, y: Tensor):
        t_embed = self.t_embedder(t).flatten(start_dim=1)
        y_embed = self.cond_embed(y)

        x = self.input_proj(x_t)
        c = t_embed + y_embed

        for adaln, lin in zip(self.ada_lns, self.linears, strict=False):
            identity = x
            modulated, gate = adaln(x, c)  # replace with c here.
            x = self.act(lin(modulated))
            x = identity + gate * x
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
        ckpt_path: str | None = None,
        method="midpoint",
        base_model_ckpt_path=None,
        beta_start_step=0,
        beta_warmup_steps=10000,
        max_beta=1.0,
        n_layers=2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.cond_dim = get_conditional_len(catalog)
        self.drop_variables = catalog["drop_variables"]
        self.lr = lr
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.beta_start_step = beta_start_step
        self.beta_warmup_steps = beta_warmup_steps
        self.max_beta = max_beta
        # --- Models --- #
        self.vf = VelocityField(code_dim, hidden_dim, self.cond_dim, n_layers)
        self.vf.apply(self._init_weights)
        self.base_model = base_model

        if base_model_ckpt_path:
            base_model_state_dict = torch.load(base_model_ckpt_path)["state_dict"]
            base_model_state_dict = {
                k.replace("vae.", "", 1): v for k, v in base_model_state_dict.items()
            }

            self.base_model.load_state_dict(base_model_state_dict)
            print("✅ Base model weights loaded.")

        # 2. Freeze the base model
        self.base_model.eval()  # Set to evaluation mode
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
            self.wrapped_vf = WrappedModel(self.vf)
            self.solver = ODESolver(velocity_model=self.wrapped_vf)

        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.method = method
        self.step_size = 1.0 / n_steps

        self.test_step_outputs = []  # To store latents and labels

    @property
    def T(self):
        return torch.tensor([1.0, 0.0], device=self.device)
        # torch.linspace(1, 0, self.n_steps, device=self.device)

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
        # beta parameters for KL loss
        # config these parameters.

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

        mu_model, log_var = self.vf.conditional_prior(y)
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
        x_t = t * x_1 + (1 - t) * x_0

        v_t = self.vf(x_t=x_t, y=y, t=t)
        v_tgt = x_1 - x_0

        cfm_loss = torch.pow(v_t - v_tgt, 2).mean()

        kl_loss = (
            0.5 * torch.sum(torch.exp(log_var) + mu_model**2 - 1 - log_var, dim=-1)
        ).mean()
        beta = self.get_beta()
        loss = cfm_loss + beta * kl_loss

        self.log(f"{partition}_loss", loss)
        self.log(f"{partition}_cfm_loss", cfm_loss)
        self.log(f"{partition}_kl_loss", kl_loss)
        self.log(f"{partition}_beta", beta)
        return loss

    def on_train_start(self):
        total_params = sum(p.numel() for p in self.vf.parameters())
        self.log("vf_total_params", total_params)
        print(f"\nModel Parameter Count: {total_params:,}")
        self.log("drop_vars_size", len(self.drop_variables))

    def training_step(self, batch, _batch_idx: int):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx: int):
        return self.base_step(batch, "val")

    def test_step(self, batch, _batch_idx: int):  # noqa: PT019
        X = batch["X"]
        y = batch["y"]

        x_1, _, _ = self.base_model.encode(X)
        batch_size = x_1.shape[0]

        mu_model, log_var = self.vf.conditional_prior(y)
        eps = torch.randn_like(x_1)
        x_0_cond = mu_model + torch.exp(0.5 * log_var) * eps

        x_0_uncond = torch.randn_like(x_1)

        x_1 = torch.cat([x_1, x_1], dim=0)
        x_0 = torch.cat([x_0_cond, x_0_uncond], dim=0)

        null_idx = torch.zeros(batch_size, dtype=torch.long, device=y.device)
        y_null = self.vf.null_y(null_idx)
        y_combined = torch.cat([y, y_null], dim=0)

        t = torch.rand(batch_size, device=x_1.device).unsqueeze(-1)
        t = torch.cat([t, t], dim=0)
        x_t = t * x_1 + (1 - t) * x_0

        v_t = self.vf(x_t=x_t, y=y_combined, t=t)
        v_tgt = x_1 - x_0

        cfm_loss = torch.pow(v_t - v_tgt, 2).mean()

        kl_loss = (
            0.5 * torch.sum(torch.exp(log_var) + mu_model**2 - 1 - log_var, dim=-1)
        ).mean()
        beta = self.get_beta()
        loss = cfm_loss + beta * kl_loss

        self.log("test_loss", loss)
        self.log("test_cfm_loss", cfm_loss)
        self.log("test_kl_loss", kl_loss)
        self.log("n_layers", self.n_layers)
        self.log("hidden_dim", self.hidden_dim)

        catalog = batch["catalog"]

        output = self.predict_step(X, y, embed_opt=["cond", "orig", "uncond"])
        output["catalog"] = {k: v.detach().cpu() for k, v in catalog.items()}
        self.test_step_outputs.append(output)
        return output  # self.base_step(batch, "test")

    def predict_step(self, X, y, embed_opt=None):
        if embed_opt is None:
            embed_opt = ["cond"]
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


if __name__ == "__main__":
    pass
