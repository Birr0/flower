import os
import time 

import torch 
from torch import Tensor

from model import MLP, Prior
from data import generate_quad_gmm

# training arguments
lr = 0.001
batch_size = 1024
iterations = 20001
print_every = 1000

target_beta = 1.0
warmup_iters = 10000

# velocity field model init
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vf = MLP(dim=2, h=64).to(device)
prior = Prior()

#vf.load_state_dict(torch.load("./vf.pth", weights_only=True))
# init optimizer
params = list(vf.parameters()) + list(prior.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

y_null_val = -1.
# train

if __name__ == "__main__":
    start_time = time.time()
    for i in range(iterations):
        optimizer.zero_grad()

        x_1, y, _, _ = generate_quad_gmm(batch_size)
        x_1 = Tensor(x_1).to(device)
        y = Tensor(y).to(device)
        y_null = torch.ones_like(y) * y_null_val

        mu_model, log_var = prior(y.unsqueeze(1).float())
        eps = torch.randn_like(x_1)
        x_0_cond = mu_model + torch.exp(0.5 * log_var) * eps
        
        x_0_uncond = torch.randn_like(x_1)

        t = torch.rand(x_1.shape[0]).to(device).unsqueeze(-1)
        t = torch.cat([t, t], dim=0)
        x_0 = torch.cat([x_0_cond, x_0_uncond], dim=0)
        x_1 = torch.cat([x_1, x_1], dim=0)
        y = torch.cat([y, y_null], dim=0)

        x_t = t*x_1 + (1 - t)*x_0
        v_t = vf(x_t=x_t, y=y, t=t)
        v_tgt = x_1 - x_0

        kl_div = 0.5 * torch.sum(torch.exp(log_var) + mu_model**2 - 1 - log_var, dim=-1)
        kl_loss = kl_div.mean()

        beta = min(target_beta, target_beta * (i/warmup_iters))
        flow_loss = torch.pow(
            v_t - v_tgt, 2
        ).mean()

        loss = flow_loss + beta*kl_loss

        # optimizer step
        loss.backward()  # backward
        optimizer.step()  # update

        # log loss
        if (i + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"| iter {i+1:6d} | {elapsed*1000/print_every:5.2f}  \
                    ms/step | loss {loss.item():8.3f} "
            )
            print(flow_loss, kl_loss)
            start_time = time.time()

    # 1. Define the folder and filename
    folder_path = './checkpoints/' # Change this to your desired folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    checkpoint_path = os.path.join(folder_path, 'cond_fm.pth')
    checkpoint = {
        'iteration': i + 1,
        'vf_state_dict': vf.state_dict(),
        'prior_state_dict': prior.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }

    # 3. Save to disk
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved successfully to {checkpoint_path}")