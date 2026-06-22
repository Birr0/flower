import torch
from torch import Tensor, nn
from flow_matching.utils import ModelWrapper

class MLP(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, t: Tensor, x_t: Tensor, y: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x_t.shape[0], 1)

        elif t.ndim == 1:
            t = t.unsqueeze(1)

        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(1)

        if y.ndim == 1:
            y = y.unsqueeze(1)

        return self.net(torch.cat((x_t, y, t), -1))

class Prior(nn.Module):
    def __init__(self, dim: int = 1, h: int = 8, data_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ELU(),
            nn.Linear(h, 2*data_dim)
        )
        # Access the final layer to initialize it
        final_layer = self.net[-1]
        
        # 1. Zero out weights so the input y doesn't affect output initially
        nn.init.zeros_(final_layer.weight)
        
        # 2. Set biases to your target values
        with torch.no_grad():
            # First half of bias is for mu, second half is for logvar
            final_layer.bias[:data_dim].fill_(0.0)    # mu = 0
            final_layer.bias[data_dim:].fill_(0.0)    # logvar = 0 (for unit variance)
    
    def forward(self, y: Tensor) -> Tensor:
        # return mu = 0, logvar=0
        mu, logvar = self.net(y).chunk(2, dim=-1)
        return mu, logvar # torch.zeros_like(mu),  torch.zeros_like(logvar) #
    
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **model_extras):
        return self.model(x_t=x, t=t, **model_extras)

class ConditionedVelocityModelWrapper(nn.Module):
    """Wrapper around velocity model to inject month condition during inference.
    Implements classifier-free guidance according to the formula:
    u ← (1-w)*u_null + w*u_cond
    where:
    - u_null is the velocity with condition dropped
    - u_cond is the velocity with condition intact
    - w is the cfg_scale (default=1.0, which means no guidance)
    """

    def __init__(self, velocity_model, cfg_scale=1.0):
        super().__init__()
        self.velocity_model = velocity_model
        self.cfg_scale = cfg_scale

    def forward(self, x, t, **model_extras):
        """Forward pass with classifier-free guidance.

        Args:
            x: Input tensor (batch_size, ...)
            t: Time tensor (batch_size, ) or ()

        Returns:
            Predicted velocity with CFG applied if cfg_scale > 1.0
        """

        y = model_extras["y"]

        if self.cfg_scale == 1.0:
            return self.velocity_model(x_t=x, t=t, y=y)

        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        v_cond = self.velocity_model(x_t=x, t=t, y=y)
        v_null = self.velocity_model(x_t=x, t=t, y=torch.ones_like(y) * -1)

        return (1 - self.cfg_scale) * v_null + self.cfg_scale * v_cond

class CVAE(nn.Module):
    def __init__(self, feature_dim=2, context_dim=1, latent_dim=2, hidden_dim=64):
        super(CVAE, self).__init__()
        
        # Encoder: (x, y) -> mu, logvar
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: (z, y) -> reconstructed_x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, y):
        # Flatten y if necessary and concat
        inputs = torch.cat([x, y], dim=1)
        h = self.encoder(inputs)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar 
    
    def forward(self, x, y):
        # Flatten y if necessary and concat
        inputs = torch.cat([x, y], dim=1)
        h = self.encoder(inputs)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        z = self.reparameterize(mu, logvar)
        
        z_cond = torch.cat([z, y], dim=1)
        recon_x = self.decoder(z_cond)
        return recon_x, mu, logvar


# training arguments
lr = 0.001
batch_size = 1024
iterations = 20001
print_every = 1000

target_beta = 1.0
warmup_iters = 10000

# velocity field model init
vf = MLP(dim=2, h=64) #.to(device)
prior = Prior()
# instantiate an affine path object
#path = AffineProbPath(scheduler=CondOTScheduler())

#vf.load_state_dict(torch.load("./vf.pth", weights_only=True))
# init optimizer
params = list(vf.parameters()) + list(prior.parameters())
optimizer = torch.optim.Adam(params, lr=lr)
y_null_val = -1.