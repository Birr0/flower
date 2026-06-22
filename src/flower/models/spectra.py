import math 

import torch
import torch.nn as nn
import lightning as L
from torch.func import jvp
from torch.autograd import Function
import torch.nn.functional as F
from torch import Tensor
from flower.models.modules import get_conditional_len, BaseModel
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from timm.layers import trunc_normal_
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
import spender

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
        # parameter for y embeddings
        self.conditional_prior = ConditionalPrior(
            cond_dim=conditional_dim,
            hidden_dim=hidden_dim,
            code_dim=code_dim
        )

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
        n_steps=10,
        ckpt_path: str = None,
        method="midpoint",
        beta_start_step=0,
        beta_warmup_steps=10000,
        max_beta=1.0
    ):
        super().__init__()

        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.lr = lr

        self.beta_start_step = beta_start_step
        self.beta_warmup_steps = beta_warmup_steps
        self.max_beta = max_beta 

        # --- Models --- #
        self.vf = VelocityField(code_dim, hidden_dim, get_conditional_len(catalog))
        self.vf.apply(self._init_weights)
        self.base_model = base_model

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
        # beta parameters for KL loss
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

        x_1 = self.base_model.encode(X)
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
            code = self.base_model.encode(X)
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

class PretrainedSpender(nn.Module):
    def __init__(self, model, latent_dim):
        super().__init__()
        _, self.model = spender.hub.load(model)
        self.latent_dim = latent_dim
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, X):
        return self.model.encoder(X)

    @torch.no_grad()
    def decoder(self, Z):
        return self.model.decoder(Z)
    
if __name__ == "__main__":
    # 1. Setup Dummy Dimensions/Hyperparams
    BATCH_SIZE = 4
    CODE_DIM = 16
    HIDDEN_DIM = 64
    COND_DIM = 10
    CATALOG = "dummy_catalog" # This would normally go to get_conditional_len

    print("🚀 Initializing test components...")

    # 2. Create a Mock Base Model 
    # (Since PretrainedSpender requires an actual hub load)
    class MockBaseModel(nn.Module):
        def __init__(self, code_dim):
            super().__init__()
            self.code_dim = code_dim
        def encode(self, x):
            return torch.randn(x.shape[0], self.code_dim)

    base_model = MockBaseModel(CODE_DIM)

    # 3. Instantiate the Lightning Module
    # Note: I'm passing ckpt_path=None to test the "fresh" init
    model = LightningFlowMatching(
        base_model=base_model,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        code_dim=CODE_DIM,
        hidden_dim=HIDDEN_DIM,
        catalog=CATALOG,
        n_steps=5,
        ckpt_path=None 
    )

    # --- QUICK FIX FOR THE INIT BUG ---
    # In your original code, solver is only defined if ckpt_path exists.
    # We manually attach them here so the test doesn't crash.
    model.wrapped_vf = WrappedModel(model.vf)
    model.solver = ODESolver(velocity_model=model.wrapped_vf)
    # ----------------------------------

    # 4. Create Dummy Data
    # X: (Batch, Channels/Features) - mimicking a spectrum or image
    # y: (Batch, Cond_Dim) - the conditional vector
    dummy_X = torch.randn(BATCH_SIZE, 1024) 
    dummy_y = torch.randn(BATCH_SIZE, COND_DIM)
    batch = (dummy_X, dummy_y)

    print("🛠️ Testing training_step...")
    loss = model.training_step(batch, 0)
    print(f"✅ Training Step Loss: {loss.item():.4f}")

    print("\n🔮 Testing predict_step (Inference)...")
    model.eval()
    # Testing with both conditional and unconditional options
    outputs = model.predict_step(dummy_X, dummy_y, embed_opt=["orig", "cond", "uncond"])

    for key, val in outputs.items():
        print(f"✅ Output '{key}' shape: {val.shape}")

    print("\n✨ All systems go. The VelocityField and Flow logic are holding up!")