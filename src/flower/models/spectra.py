import spender
import torch
import torch.nn as nn

from flower.models.modules import (
    LightningFlowMatching as LightningFlowMatchingBase,
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
