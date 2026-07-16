import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flower.models.modules import (
    LightningFlowMatching as LightningFlowMatchingBase,
)


class LightningFlowMatching(LightningFlowMatchingBase):
    pass


### --- VAE --- ###
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
        return {"z": z, "mu": mu, "logvar": log_var}

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
            state_dict = torch.load(vae_ckpt_path)["state_dict"]
            state_dict = {k.replace("vae.", "", 1): v for k, v in state_dict.items()}
            self.vae.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.vae.parameters(), lr=self.lr)

    def base_step(self, batch, partition):
        output = self.vae(batch["X"])
        if isinstance(output, tuple):
            recon, _, mu, logvar = output
            log_var = logvar
        else:
            recon = output["recon"]
            mu = output["mu"]
            log_var = output["log_var"]
        recon_loss = self.alpha * self.mse(recon, batch["X"], reduction="mean")

        kl_loss = torch.sum(
            -0.5 * (1 + log_var - log_var.exp() - mu.pow(2)),
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
