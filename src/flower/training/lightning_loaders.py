import lightning as L  # type: ignore[import-not-found]
import pyro.distributions as dists
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from zuko.distributions import DiagNormal
from zuko.lazy import UnconditionalDistribution


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
            state_dict = torch.load(
                vae_ckpt_path  # ,
                # map_location="cpu"
            )["state_dict"]
            state_dict = {k.replace("vae.", "", 1): v for k, v in state_dict.items()}
            self.vae.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.vae.parameters(), lr=self.lr)

    def base_step(self, batch, partition):
        output = self.vae(batch["X"])
        # z = self.vae.reparametrize(output["mu"], output["log_var"])
        recon_loss = self.alpha * self.mse(
            output["recon"], batch["X"], reduction="mean"
        )

        kl_loss = torch.sum(
            -0.5
            * (1 + output["log_var"] - output["log_var"].exp() - output["mu"].pow(2)),
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

        print(output["z"].shape)

        return {
            "X": batch["X"],
            "recon": output["recon"],
            "z": output["z"].flatten(start_dim=1),
            "catalog": batch["catalog"],
            "label": batch["label"],
        }


class LightningFlow(L.LightningModule):
    def __init__(
        self,
        flow1,
        flow2,
        vae,
        lr,
        batch_size,
        size,
        conditional_size,
        vae_ckpt_path,
        ckpt_path=None,
    ):
        super().__init__()
        self.flow1 = flow1
        self.flow2 = flow2
        self.vae = vae

        state_dict = torch.load(
            vae_ckpt_path  # ,
            # map_location="cpu"
        )["state_dict"]

        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("vae.")  # _orig_mod
        }

        state_dict = {
            k.replace("vae.", ""): v for k, v in state_dict.items()
        }  # _orig_mod

        self.vae.load_state_dict(state_dict, strict=False)
        self.vae.eval()

        if ckpt_path:
            flow_state_dict = torch.load(
                ckpt_path  # ,
                # map_location="cpu"
            )["state_dict"]

            # This needs a more elegant solution than
            # key checking.

            flow1_state_dict = {
                k: v for k, v in flow_state_dict.items() if k.startswith("flow1.")
            }

            flow2_state_dict = {
                k: v for k, v in flow_state_dict.items() if k.startswith("flow2.")
            }

            flow1_state_dict = {
                k.replace("flow1.", "", 1): v  # _orig_mod.
                for k, v in flow1_state_dict.items()
            }

            flow2_state_dict = {
                k.replace("flow2.", "", 1): v  # _orig_mod.
                for k, v in flow2_state_dict.items()
            }

            self.flow1.load_state_dict(flow1_state_dict)
            self.flow2.load_state_dict(flow2_state_dict)

        else:
            self.flow1.apply(self._init_weights)
            self.flow2.apply(self._init_weights)

        self.lr = lr
        self.batch_size = batch_size
        self.size = size  # latent dim

        self.prior = DiagNormal(
            loc=torch.zeros(size, device=self.device),
            scale=torch.ones(size, device=self.device),
        )
        self.null_y = nn.Embedding(1, conditional_size, device=self.device)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.flow2.parameters())
            + list(self.flow1.parameters())
            + list(self.null_y.parameters()),
            lr=self.lr,
        )

    def base_step(self, batch, partition):
        # print each variable used and their device.
        output = self.vae(batch["X"])
        z = output["z"]
        y = batch["y"]

        # mu, var = self.bn(output["mu"]), output["log_var"].exp()

        """self.flow1.base = UnconditionalDistribution(
            DiagNormal,
            mu,
            var
        )"""

        self.flow2.base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(self.size, device=z.device),
            torch.ones(self.size, device=z.device),
        )

        # uncondz_ll = self.flow2(y).log_prob(z)/self.size

        mask = torch.rand(y.size(0), 1, device=y.device)

        y = torch.where(
            mask < 0.2,
            self.null_y(torch.zeros(y.size(0), dtype=torch.long, device=y.device)),
            y,
        )

        self.flow2.base = self.flow1
        ll = self.flow2(y).log_prob(z)

        z_prime = self.flow1(y).sample()
        kl = (self.flow1(y).log_prob(z_prime) - self.prior.log_prob(z_prime)).mean()
        loss = -ll.mean()  # + kl.mean()

        self.log(f"{partition}_nll", -ll.mean())

        # self.log(f"{partition}_kl_loss", kl.mean())
        self.log(f"{partition}_kl_loss", kl.mean())
        self.log(f"{partition}_loss", loss)
        return loss

    def training_step(self, batch):
        return self.base_step(batch, "train")

    def validation_step(self, batch):
        return self.base_step(batch, "val")

    def test_step(self, batch):
        return self.base_step(batch, "test")

    def predict_step(self, batch):
        output = self.vae(batch["X"])
        mu, var = output["mu"], output["log_var"].exp()
        self.flow1.base = UnconditionalDistribution(DiagNormal, mu, var)

        z_prime = self.flow1(output["y"]).sample()

        return {
            "z": self.flow1(output["y"]).base.sample(),
            "z_prime": z_prime,
            "y": batch["y"],
            "catalog": batch["catalog"],
        }


class LightningScheduleVAE(L.LightningModule):
    def __init__(self, model, optimizers_config, vae_ckpt_path=None, ckpt_path=None):
        super().__init__()
        self.model = model  # torch.compile()
        self.optimizers_config = optimizers_config
        self.ckpt_path = ckpt_path
        # this is not needed but required for consistency.
        # in other functions. Think of better fix.

        if vae_ckpt_path:
            checkpoint = torch.load(vae_ckpt_path)
            state_dict = {
                k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
            }
            self.model.load_state_dict(state_dict)

        self.loss_factors = {
            "kl_divergence": torch.tensor(1.0),
            "target": torch.tensor(1.0),
            "recon": torch.tensor(1.0),
        }

    @staticmethod
    def recon_loss(recon, target):
        return torch.nn.MSELoss()(recon, target)

    @staticmethod
    def kl_loss(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def supervised_losses(self, output, batch):
        loss_size = len(self.model.targets)
        losses = torch.empty(loss_size)
        weights = torch.empty(loss_size)
        # metrics = {}

        for i, (var_name, variables) in enumerate(self.model.targets.items()):
            if variables["supervised_prediction"]:
                losses[i] = variables["supervised_prediction"]["loss_fn"](
                    output["targets"][var_name], batch["catalog"][var_name]
                )
                weights[i] = variables["supervised_prediction"]["weight"]

                """metrics[var_name] = {
                    metric: variables["supervised_prediction"]["metrics"][metric](
                        output["targets"][var_name].detach().cpu(),
                        batch["catalog"][var_name].detach().cpu()
                    )
                    for metric in variables["supervised_prediction"]["metrics"]
                }"""

        return {
            # "metrics": metrics,
            "losses": losses,
            "weights": weights,
        }

    def base_step(self, batch, partition, schedulers=None):
        output = self.model(batch["X"])

        if schedulers is not None:
            for scheduler in schedulers:
                self.loss_factors[scheduler.name] = scheduler.annealing_factor

        recon_loss = self.loss_factors["recon"] * self.recon_loss(
            output["recon"], batch["X"]
        )
        kl_loss = self.loss_factors["kl_divergence"] * self.kl_loss(
            output["mu"], output["log_var"]
        )
        supervised_losses = self.supervised_losses(output, batch)
        wsup_loss = self.loss_factors["target"] * torch.sum(
            supervised_losses["losses"] * supervised_losses["weights"]
        )

        total_loss = recon_loss + kl_loss + wsup_loss

        self.log(f"{partition}_recon_loss", recon_loss)
        self.log(f"{partition}_kl_loss", kl_loss)
        self.log(f"{partition}_supervised_loss", wsup_loss)
        self.log(f"{partition}_total_loss", recon_loss + kl_loss)

        """for var_name, variables in supervised_losses["metrics"].items():
            for metric_name, val in variables.items():
                self.log(f"{partition}_{var_name}_{metric_name}", val)"""

        return total_loss

    def training_step(self, batch):
        schedulers = self.lr_schedulers()
        total_loss = self.base_step(batch, "train", schedulers)
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        else:
            if schedulers is not None:
                schedulers.step()
        return total_loss

    def validation_step(self, batch):
        return self.base_step(batch, "val")

    def test_step(self, batch):
        return self.base_step(batch, "test")

    def predict_step(self, batch):
        output = self.model(batch["X"])

        output["targets"] = {
            f"{k}_pred": v for k, v in output["targets"].items() if k != "y"
        }

        return {
            "X": batch["X"],
            # "recon": output["recon"],
            "z": output["z"],
            "z_prime": output["z_prime"],
            # "y": batch["y"],
            "catalog": batch["catalog"],
            **output["targets"],
        }

    def configure_optimizers(self):
        optims = []
        lr_schedulers = []

        for optimizer in self.optimizers_config:
            params = []

            if optimizer["optim_components"] == "all":
                params = self.model.parameters()
            else:
                for component in optimizer["optim_components"]:
                    params.extend(
                        getattr(self.model.components, component).parameters()
                    )

            optim = optimizer.target(
                params,
                **optimizer["optim_args"],
            )

            optims.append(optim)

            hparam_schedulers = optimizer["hparam_schedulers"]

            for name, hparam_scheduler in hparam_schedulers.items():
                lr_schedulers.append(
                    {
                        "name": name,
                        "scheduler": hparam_scheduler(optim),
                        "interval": "step",  # or 'epoch'
                        "frequency": 1,
                    }
                )

        return optims, lr_schedulers


class CategoricalModel(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(n_in, n_out))
        self.distribution = dists.OneHotCategoricalStraightThrough
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

    def sample(self, x):
        return self.distribution(
            logits=self.forward(x),
        ).sample()

    def log_prob(self, x, y):
        if y.size(1) == 1 and len(y.shape) == 3:
            # Reduce the expanded one hot encoding to 2D
            y = y.squeeze(1)
        return self.distribution(
            logits=self.forward(x),
        ).log_prob(y)


class GaussianModel(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.distribution = dists.Normal
        self.model = nn.Sequential(
            nn.Linear(n_in, 2 * n_out),
        )
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x).chunk(2, dim=-1)

    def sample(self, x):
        mu, logvar = self.forward(x)
        return self.distribution(
            loc=mu,
            scale=logvar.exp(),
        ).sample()

    def log_prob(self, x, y):
        mu, logvar = self.forward(x)
        return self.distribution(
            loc=mu,
            scale=logvar.exp(),
        ).log_prob(y)


class LightningLikelihood(L.LightningModule):
    def __init__(self, X, y, model, lr, model_ckpt_path=None, name=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.X = X
        self.y = y
        self.name = name

        if model_ckpt_path:
            state_dict = torch.load(model_ckpt_path)["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.vae.load_state_dict(state_dict)
            self.vae.eval()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def base_step(self, batch, partition):
        loss = -self.model.log_prob(batch[self.X], batch[self.y]).mean()
        self.log(f"{partition}_loss", loss)
        return loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch):
        return self.base_step(batch, "test")

    def predict_step(self, batch):
        y_samples = self.model.sample(batch[self.X])

        return {
            "X": batch[self.X],
            "y": batch[self.y],
            "y_pred": y_samples,
            "log_prob": self.model.log_prob(batch[self.X], batch[self.y]),
            **batch,
        }
