import lightning as L
import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class LightningModel(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super().__init__()
        self.model = model
        self.lr = lr

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )

    def training_step(self, _batch, _batch_idx: int) -> None:
        # training logic
        return

    def validation_step(self, _batch, _batch_idx: int) -> None:
        # val logic
        return

    def test_step(self, _batch, _batch_idx: int) -> None:  # noqa: PT019
        # test logic
        return

    def predict_step(self, _batch) -> None:
        # inference logic
        return
