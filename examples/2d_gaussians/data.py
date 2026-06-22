import torch
import torch.distributions as D

weight = 0.25
mixture_distribution = D.Categorical(torch.tensor(4 * [weight]))
mu_data = torch.tensor([[3, 3], [-3, 3], [-3, -3], [3, -3]], dtype=torch.float32)
covariances = torch.tensor(
    [
        [[0.5, 0.0], [0.0, 0.5]],
        [[0.5, 0.0], [0.0, 0.5]],
        [[0.5, 0.0], [0.0, 0.5]],
        [[0.5, 0.0], [0.0, 0.5]],
    ],
    dtype=torch.float32,
)

def generate_quad_gmm(num_samples):
    component_indices = mixture_distribution.sample((num_samples,))
    selected_means = torch.stack([mu_data[idx, :] for idx in component_indices])
    selected_covs = torch.stack([covariances[idx, :, :] for idx in component_indices])
    selected_components = D.MultivariateNormal(
        loc=selected_means, covariance_matrix=selected_covs
    )
    samples = selected_components.sample()
    diff = samples - selected_means
    return samples, component_indices, torch.sqrt(torch.sum(diff**2, dim=-1)), diff