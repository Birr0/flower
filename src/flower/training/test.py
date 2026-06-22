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


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model,
        lr
    ):
        super().__init__()
        self.model = model 
        self.lr = lr

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )

    def training_step(self, batch, _batch_idx):
        # training logic
        return

    def validation_step(self, batch, _batch_idx):
        # val logic
        return 

    def test_step(self, batch, _batch_idx):
        # test logic
        return

    def predict_step(self, batch):
        # inference logic
        return
