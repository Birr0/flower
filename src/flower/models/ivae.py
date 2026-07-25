"""Identifiable VAE (iVAE) — nonlinear-ICA baseline.

Follows the reference implementation https://github.com/ilkhem/iVAE (Khemakhem,
Kingma, Monti, Hyvärinen, "Variational Autoencoders and Nonlinear ICA: A
Unifying Framework", AISTATS 2020, arXiv:1907.04809). The architecture mirrors
their ``models.nets.iVAE``:

- ``MLP`` blocks with the **xtanh** activation (``tanh(x) + slope*x``) and
  Xavier-uniform init.
- encoder ``q(s | x, u) = N(g(x,u), diag exp(logv(x,u)))``,
- decoder ``p(x | s) = N(f(s), decoder_var * I)`` with a **fixed** small
  ``decoder_var`` (0.01) — this is what upweights reconstruction, not an ad-hoc
  loss weight,
- conditional prior ``p(s | u) = N(mu_p(u), diag exp(logl(u)))``. The reference's
  synthetic ``iVAE`` fixes the prior mean to 0 (its benchmark sources are
  zero-mean, variance-modulated); their MNIST variant modulates the mean too. We
  keep both via ``learn_prior_mean`` — the conditional prior is exactly what makes
  the sources identifiable and localises the ``u``-dependence.

This is a *baseline* for Flower, not part of the main model. To keep it
apples-to-apples with the other residual baselines, it operates on a flat
feature vector ``x`` (a pretrained embedding), with the auxiliary variable ``u``
being the (embedded) conditioning ``y``.

Encoder contract matches the rest of ``flower.models.*``: ``encode`` returns
``{"z": s, "mu": mu, "logvar": logvar}``.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class MLP(nn.Module):
    """MLP with a configurable activation, matching ilkhem/iVAE ``models.nets.MLP``.

    The activation is applied after every layer except the last. ``n_layers``
    counts total linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 100,
        n_layers: int = 3,
        activation: str = "xtanh",
        slope: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.slope = slope
        self.activation = activation

        if n_layers == 1:
            dims = [input_dim, output_dim]
        else:
            dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.fc = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)
        )

    def _act(self, x: Tensor) -> Tensor:
        if self.activation == "xtanh":
            return x.tanh() + self.slope * x
        if self.activation == "lrelu":
            return nn.functional.leaky_relu(x, negative_slope=self.slope)
        if self.activation == "none":
            return x
        msg = f"incorrect activation: {self.activation}"
        raise ValueError(msg)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.fc):
            x = layer(x)
            if i < self.n_layers - 1:
                x = self._act(x)
        return x


class IVAE(nn.Module):
    """Factorised-Gaussian iVAE over a feature vector with auxiliary ``u``.

    Args:
        data_dim: dimensionality of the input feature vector ``x``.
        aux_dim: dimensionality of the auxiliary variable ``u`` (the condition).
        latent_dim: number of latent sources ``s`` (defaults to ``data_dim``).
        hidden_dim: width of every MLP.
        n_layers: linear layers per MLP.
        activation: MLP activation (``"xtanh"`` in the reference).
        slope: slope parameter for the activation.
        decoder_var: fixed decoder (observation) variance.
        learn_prior_mean: if True, the conditional prior mean is a function of
            ``u`` (as in the reference MNIST iVAE); if False the prior mean is
            fixed at 0 (as in the reference synthetic iVAE).
        condition_encoder: if True (the reference iVAE) the encoder sees
            ``[x, u]``; if False it sees only ``x``. Conditioning on ``u`` is what
            gives iVAE its identifiability, but when the model is used as a
            *residualiser* to remove ``u`` it lets the sources bake ``u`` back in
            (a nonlinear probe then recovers it); ``condition_encoder=False`` is
            the fair removal variant (a conditional-prior VAE — the ``u``-signal
            enters only through the prior).
    """

    def __init__(
        self,
        data_dim: int,
        aux_dim: int,
        latent_dim: int | None = None,
        hidden_dim: int = 100,
        n_layers: int = 3,
        activation: str = "xtanh",
        slope: float = 0.1,
        decoder_var: float = 0.01,
        learn_prior_mean: bool = True,
        condition_encoder: bool = True,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.aux_dim = aux_dim
        self.latent_dim = latent_dim or data_dim
        self.learn_prior_mean = learn_prior_mean
        self.condition_encoder = condition_encoder

        mlp = lambda i, o: MLP(i, o, hidden_dim, n_layers, activation, slope)  # noqa: E731

        # encoder q(s|x,u) (or q(s|x) if not condition_encoder): mean/logvar heads
        enc_in = data_dim + aux_dim if condition_encoder else data_dim
        self.g = mlp(enc_in, self.latent_dim)
        self.logv = mlp(enc_in, self.latent_dim)
        # decoder p(x|s)
        self.f = mlp(self.latent_dim, data_dim)
        # conditional prior p(s|u): log-variance always; mean optional
        self.logl = mlp(aux_dim, self.latent_dim)
        self.prior_mean_net = (
            mlp(aux_dim, self.latent_dim) if learn_prior_mean else None
        )

        self.register_buffer("decoder_var", torch.tensor(float(decoder_var)))
        self.apply(_weights_init)

    @staticmethod
    def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def prior_params(self, u: Tensor) -> tuple[Tensor, Tensor]:
        lam_logvar = torch.clamp(self.logl(u), -30.0, 20.0)
        if self.prior_mean_net is not None:
            lam_mu = self.prior_mean_net(u)
        else:
            lam_mu = torch.zeros_like(lam_logvar)
        return lam_mu, lam_logvar

    def encode(self, x: Tensor, u: Tensor) -> dict[str, Tensor]:
        xu = torch.cat([x, u], dim=-1) if self.condition_encoder else x
        mu = self.g(xu)
        logvar = torch.clamp(self.logv(xu), -30.0, 20.0)
        s = self.reparametrize(mu, logvar)
        return {"z": s, "mu": mu, "logvar": logvar}

    def forward(self, x: Tensor, u: Tensor) -> dict[str, Tensor]:
        enc = self.encode(x, u)
        recon = self.f(enc["z"])
        lam_mu, lam_logvar = self.prior_params(u)
        return {
            "recon": recon,
            "z": enc["z"],
            "mu": enc["mu"],
            "logvar": enc["logvar"],
            "prior_mu": lam_mu,
            "prior_logvar": lam_logvar,
        }


class LightningIVAE(L.LightningModule):
    """Lightning wrapper training the iVAE ELBO with the conditional prior.

    Loss = Gaussian reconstruction NLL (fixed ``decoder_var``) + ``beta`` * the
    closed-form KL to the *conditional* prior p(s|u). ``beta=1`` is the standard
    ELBO; the reconstruction weighting comes from the small ``decoder_var``, as
    in the reference. Default ``lr=1e-2`` matches ilkhem/iVAE.
    """

    def __init__(
        self,
        ivae: IVAE,
        lr: float = 1e-2,
        batch_size: int = 64,
        beta: float = 1.0,
        ckpt_path: str | None = None,
    ):
        super().__init__()
        self.ivae = ivae
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.ckpt_path = ckpt_path

    def configure_optimizers(self):
        # Adam + ReduceLROnPlateau matches ilkhem/iVAE; the LR decay is important
        # for the encoder to converge to an identifying solution.
        opt = torch.optim.Adam(self.ivae.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.1, patience=3
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }

    @staticmethod
    def _kl_to_prior(
        mu: Tensor, logvar: Tensor, prior_mu: Tensor, prior_logvar: Tensor
    ) -> Tensor:
        """Closed-form KL(N(mu, s_q^2) || N(prior_mu, s_p^2)), summed over dims,
        averaged over the batch."""
        var_q = logvar.exp()
        var_p = prior_logvar.exp()
        kl = 0.5 * (
            prior_logvar - logvar + (var_q + (mu - prior_mu).pow(2)) / var_p - 1.0
        )
        return kl.sum(dim=-1).mean()

    def _recon_nll(self, x: Tensor, recon: Tensor) -> Tensor:
        """Gaussian negative log-likelihood with fixed decoder variance, summed
        over data dims and averaged over the batch."""
        var = self.ivae.decoder_var
        nll = 0.5 * ((x - recon).pow(2) / var + torch.log(2 * np.pi * var))
        return nll.sum(dim=-1).mean()

    def _losses(self, out: dict[str, Tensor], x: Tensor):
        recon_loss = self._recon_nll(x, out["recon"])
        kl_loss = self._kl_to_prior(
            out["mu"], out["logvar"], out["prior_mu"], out["prior_logvar"]
        )
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def base_step(self, batch, partition: str) -> Tensor:
        x, u = batch["X"], batch["u"]
        loss, recon_loss, kl_loss = self._losses(self.ivae(x, u), x)
        self.log(f"{partition}_loss", loss, sync_dist=True)
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
        x, u = batch["X"], batch["u"]
        out = self.ivae(x, u)
        return {
            "X": x,
            "z": out["z"],
            "prior_mu": out["prior_mu"],
            "catalog": batch.get("catalog"),
        }
