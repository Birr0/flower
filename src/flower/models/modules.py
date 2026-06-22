from typing import Protocol
import math 

import torch
import torch.nn as nn 
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
    continuous_size = sum(y_catalog["variables"][v]["continuous"] for v in y_catalog["variables"])
    drop_size = sum(
        y_catalog["variables"][v]["continuous"] for v in y_catalog["drop_variables"]
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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class AdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # We predict scale (gamma) and shift (beta)
        self.ada_lin = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3*hidden_dim,  bias=True)
        )

    def forward(self, x, cond_emb):
        # Generate params from conditioning
        gamma, beta, gate = self.ada_lin(cond_emb).chunk(3, dim=-1)
    
        # Apply normalization first, then scale and shift
        x = self.norm(x)
        x = x * (1 + gamma) + beta # (1 + gamma) helps initialization stay near identity
        return self.linear(x), gate # return gating mechanism

class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, cond_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Linear(
            cond_dim, hidden_size
        )

    def forward(self, y):
        return self.embedding(y)

class ConditionalPrior(nn.Module):
    def __init__(
        self, 
        cond_dim: int, 
        hidden_dim: int, 
        code_dim: int
    ):
        super().__init__()
        # self.code_dim = code_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2*code_dim)
        )
        self._init_weights()

    def _init_weights(self):
        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias) 
    
    def forward(self, y: Tensor) -> Tensor:
        # return mu = 0, logvar=0
        mu, logvar = self.net(y).chunk(2, dim=-1)
        return mu, logvar
    
if __name__ == "__main__":
    pass