import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.solver import ODESolver
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from flower.models.modules import (
    LightningFlowMatching as LightningFlowMatchingBase,
)
from flower.models.modules import (
    WrappedModel,
)


class LightningFlowMatching(LightningFlowMatchingBase):
    def on_test_start(self):
        self.wrapped_vf = WrappedModel(self.vf)
        self.solver = ODESolver(velocity_model=self.wrapped_vf)

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
        return output

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        # 1. Aggregate Latents
        latent_map = {
            "cond": torch.cat(
                [x["cond"].detach().cpu() for x in self.test_step_outputs], dim=0
            ).numpy(),
            "uncond": torch.cat(
                [x["uncond"].detach().cpu() for x in self.test_step_outputs], dim=0
            ).numpy(),
            "orig": torch.cat(
                [x["orig"].detach().cpu() for x in self.test_step_outputs], dim=0
            ).numpy(),
        }

        # 2. Aggregate all catalog keys
        catalog_keys = self.test_step_outputs[0]["catalog"].keys()
        all_targets = {}
        for k in catalog_keys:
            all_targets[k] = torch.cat(
                [x["catalog"][k] for x in self.test_step_outputs], dim=0
            ).numpy()

        print(
            f"\\n--- Running MLP Probes on {len(catalog_keys)} attributes "
            "for 3 latent types ---"
        )

        # 3. Outer loop: Iterate through the three types of embeddings
        for latent_name, latent_vals in latent_map.items():
            print(f"\nProbing Latent Type: {latent_name}")

            # Inner loop: Iterate through each attribute in the catalog
            for key, y_vals in all_targets.items():
                # Determine if this specific key is Regression or Classification
                is_regression = key != "label_shape"

                # 4. Split values into 80% train / 20% test for the probe
                z_train, z_test, y_train, y_test = train_test_split(
                    latent_vals, y_vals, test_size=0.20, random_state=42
                )

                if is_regression:
                    # MLP Regressor (No scaling as requested)
                    probe = MLPRegressor(
                        hidden_layer_sizes=(64,),
                        max_iter=1000,
                        early_stopping=True,
                        random_state=42,
                    )
                    probe.fit(z_train, y_train)
                    score = probe.score(
                        z_test, y_test
                    )  # R^2 score on the 20% test split
                    metric_name = f"test_probe_{latent_name}_{key}_r2"
                    unit = "R2"
                else:
                    # MLP Classifier (No scaling as requested)
                    probe = MLPClassifier(
                        hidden_layer_sizes=(64,),
                        max_iter=1000,
                        early_stopping=True,
                        random_state=42,
                    )
                    probe.fit(z_train, y_train.astype(int))
                    # Accuracy on the 20% test split
                    score = accuracy_score(y_test.astype(int), probe.predict(z_test))
                    metric_name = f"test_probe_{latent_name}_{key}_acc"
                    unit = "Acc"

                # 5. Log and Print
                self.log(metric_name, score, prog_bar=False)
                print(f"  [{latent_name}] Attribute '{key}': {score:.4f} ({unit})")

        # Clear memory to prevent leak in next test run
        self.test_step_outputs.clear()


### ---- VAE ---- ###
class ResidualEncoderBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        nonlin: nn.Module | None = None,
        kernel_size=3,
        block_type="cabd",
        dropout=0.1,
        stride=2,
    ):
        super().__init__()
        self.nonlin = nonlin if nonlin is not None else nn.ReLU()
        self.pre_conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
        )

        res = []
        for character in block_type:
            if character == "a":
                res.append(self.nonlin)
            elif character == "b":
                res.append(nn.BatchNorm2d(c_out))
            elif character == "c":
                res.append(
                    nn.Conv2d(
                        c_out, c_out, kernel_size=kernel_size, padding=kernel_size // 2
                    )
                )
            elif character == "d":
                res.append(nn.Dropout2d(dropout))

        self.res = nn.Sequential(*res)

    def forward(self, x):
        x = self.pre_conv(x)
        return (self.res(x) + x).contiguous()


class ResidualDecoderBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        nonlin: nn.Module | None = None,
        kernel_size=3,
        block_type="cabd",
        dropout=0.1,
        stride=2,
        output_padding: int = 1,
    ):
        super().__init__()
        self.nonlin = nonlin if nonlin is not None else nn.ReLU()
        self.pre_conv = nn.ConvTranspose2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            output_padding=output_padding,
        )

        res = []
        for character in block_type:
            if character == "a":
                res.append(self.nonlin)
            elif character == "b":
                res.append(nn.BatchNorm2d(c_out))
            elif character == "c":
                res.append(
                    nn.ConvTranspose2d(
                        c_out, c_out, kernel_size=kernel_size, padding=kernel_size // 2
                    )
                )
            elif character == "d":
                res.append(nn.Dropout2d(dropout))

        self.res = nn.Sequential(*res)

    def forward(self, x):
        x = self.pre_conv(x)
        return (self.res(x) + x).contiguous()


# --- Main VAE Architecture ---


class BetaVAE(nn.Module):
    def __init__(
        self,
        filters: list[int] | None = None,
        latent_dim: int = 10,
        block_type: str = "cabd",
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.filters = filters if filters is not None else [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.img_dim = 64

        # 1. Build Encoder
        enc_layers = []
        in_channels = 1
        filters_arg = filters if filters is not None else [32, 64, 128, 256]
        for f in filters_arg:
            enc_layers.append(
                ResidualEncoderBlock(
                    in_channels, f, block_type=block_type, dropout=drop_rate
                )
            )
            in_channels = f
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Calculate shape after convolutions
        # Each ResidualEncoderBlock has stride 2
        self.final_feat_dim = self.img_dim // (2 ** len(filters_arg))
        self.conv_out_dim = (self.final_feat_dim**2) * filters_arg[-1]

        # 2. Latent Space
        self.fc_mu = nn.Linear(self.conv_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_out_dim, latent_dim)

        # 3. Build Decoder
        self.decoder_input = nn.Linear(latent_dim, self.conv_out_dim)

        dec_layers = []
        rev_filters = list(reversed(filters_arg))
        for i in range(len(rev_filters) - 1):
            out_pad = 1
            dec_layers.append(
                ResidualDecoderBlock(
                    rev_filters[i],
                    rev_filters[i + 1],
                    block_type=block_type,
                    dropout=drop_rate,
                    output_padding=out_pad,
                )
            )

        # Final layer to bring back to 1 channel (Logits)
        dec_layers.append(
            nn.ConvTranspose2d(rev_filters[-1], 1, kernel_size=4, stride=2, padding=1)
        )
        self.decoder_conv = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # Encode
        h = self.encoder_conv(x)
        h = torch.flatten(h, 1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        # Sample
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        # Encode
        h = self.encoder_conv(x)
        h = torch.flatten(h, 1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        h_dec = self.decoder_input(z)
        h_dec = h_dec.view(
            -1, self.filters[-1], self.final_feat_dim, self.final_feat_dim
        )
        reconstruction = self.decoder_conv(h_dec)

        # IMPORTANT: Return raw logits for F.binary_cross_entropy_with_logits
        return reconstruction, z, mu, logvar


# --- Lightning Wrapper ---


class LightningBetaVAE(L.LightningModule):
    def __init__(self, vae, lr=1e-4, beta=1.0, alpha=1.0, vae_ckpt_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=["vae"])
        self.vae = vae
        self.lr = lr
        self.beta = beta
        self.alpha = alpha

        if vae_ckpt_path:
            self._load_internal_vae(vae_ckpt_path)

    def _load_internal_vae(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt["state_dict"]
        new_state_dict = {k.replace("vae.", "", 1): v for k, v in state_dict.items()}
        self.vae.load_state_dict(new_state_dict, strict=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr)

    def base_step(self, batch, partition):
        x = batch["X"]
        # recon are logits
        recon, _, mu, logvar = self.vae(x)

        # Reconstruction Loss (Using BCE with Logits for stability)
        recon_loss = (
            F.binary_cross_entropy_with_logits(recon, x, reduction="sum") / x.shape[0]
        )

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        total_loss = (self.alpha * recon_loss) + (self.beta * kl_loss)

        self.log(f"{partition}_loss", total_loss, prog_bar=True)
        self.log(f"{partition}_kl", kl_loss)
        self.log(f"{partition}_recon", recon_loss)

        return total_loss

    def training_step(self, batch, _idx: int):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _idx: int):
        return self.base_step(batch, "val")

    def test_step(self, batch, _idx: int):  # noqa: PT019
        return self.base_step(batch, "test")

    def predict_step(self, batch):
        recon_logits, z, _, _ = self.vae(batch["X"])
        return {
            "X": batch["X"],
            "recon": torch.sigmoid(
                recon_logits
            ),  # Apply sigmoid here for visualization
            "z": z,
        }


if __name__ == "__main__":
    # 1. Setup Hyperparameters & Dummy Data
    B, C, H, W = 4, 1, 64, 64  # Batch, Channels, Height, Width
    latent_dim = 10
    hidden_dim = 64
    cond_dim = 5
    dummy_catalog = {"some_var": 5}  # Mock catalog for get_conditional_len

    # Create Dummy Inputs
    dummy_x = torch.randn(B, C, H, W)
    dummy_y = torch.randn(B, cond_dim)
    dummy_batch = (dummy_x, dummy_y)

    print(f"🚀 Starting Shape Tests (Batch Size: {B})")
    print("-" * 30)

    # 2. Test VAE
    vae = BetaVAE(latent_dim=latent_dim)
    recon, z, mu, logvar = vae(dummy_x)

    print(f"VAE Input:  {dummy_x.shape}")
    print(f"VAE Recon:  {recon.shape}  (Expected: [B, 1, 64, 64])")
    print(f"VAE Latent: {z.shape}      (Expected: [B, {latent_dim}])")

    assert recon.shape == dummy_x.shape
    assert z.shape == (B, latent_dim)

    # 3. Test Flow Matching Module
    # Note: We pass the VAE as the base_model
    model = LightningFlowMatching(
        base_model=vae,
        lr=1e-4,
        batch_size=B,
        code_dim=latent_dim,
        hidden_dim=hidden_dim,
        catalog=dummy_catalog,
    )

    # Test Training Step Shape (Loss is a scalar)
    loss = model.training_step(dummy_batch, 0)
    print(f"Flow Matching Loss: {loss.item():.4f} (Scalar expected)")

    assert loss.dim() == 0

    # 4. Test Velocity Field (Internal)
    # We test the doubled batch logic used in base_step
    t = torch.rand(B)
    v_out = model.vf(x_t=z, t=t, y=dummy_y)
    print(f"Velocity Field Output: {v_out.shape} (Expected: [B, {latent_dim}])")

    assert v_out.shape == (B, latent_dim)

    # 5. Test Inference/Predict Step
    # We need to initialize the solver manually since no ckpt was provided
    model.wrapped_vf = WrappedModel(model.vf)
    model.solver = ODESolver(velocity_model=model.wrapped_vf)

    out_dict = model.predict_step(dummy_x, dummy_y, embed_opt=["orig", "cond"])

    print(f"Predict 'orig' shape: {out_dict['orig'].shape}")
    print(f"Predict 'cond' shape: {out_dict['cond'].shape}")

    print("-" * 30)
    print("✅ All shape tests passed!")
