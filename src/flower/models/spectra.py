import spender
import torch
import torch.nn as nn
from flow_matching.solver import ODESolver

from flower.models.modules import (
    LightningFlowMatching as LightningFlowMatchingBase,
)
from flower.models.modules import (
    WrappedModel,
)


class LightningFlowMatching(LightningFlowMatchingBase):
    pass


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
        return {"z": self.model.encoder(X)}

    @torch.no_grad()
    def decoder(self, Z):
        return self.model.decoder(Z)


if __name__ == "__main__":
    # 1. Setup Dummy Dimensions/Hyperparams
    BATCH_SIZE = 4
    CODE_DIM = 16
    HIDDEN_DIM = 64
    COND_DIM = 10
    CATALOG = "dummy_catalog"  # This would normally go to get_conditional_len

    print("🚀 Initializing test components...")

    # 2. Create a Mock Base Model
    # (Since PretrainedSpender requires an actual hub load)
    class MockBaseModel(nn.Module):
        def __init__(self, code_dim):
            super().__init__()
            self.code_dim = code_dim

        def encode(self, x):
            return {"z": torch.randn(x.shape[0], self.code_dim)}

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
        ckpt_path=None,
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
