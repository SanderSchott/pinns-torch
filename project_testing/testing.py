from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import lightning.pytorch as pl

import pinnstorch
import torch
from torch import nn
from lightning import LightningModule
from typing import Dict, Tuple, List, Any, Union

class FCNLightning(LightningModule):
    def __init__(
        self,
        fcn: nn.Module,
        lr: float = 1e-3,
        optimizer_class: Any = torch.optim.Adam,
        scheduler_class: Any = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["fcn"])
        self.model = fcn
        self.loss_fn = nn.MSELoss()

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor):
        return self.model(spatial, time)

    def _compute_batch_loss(self, batch):
        loss = torch.tensor(0.0, device=self.device)

        # support both dict‐batches and (spatial, time, u_true) tuples…
        if isinstance(batch, dict):
            iterator = batch.items()
        else:
            iterator = [(None, batch)]

        for cond, data in iterator:
            spatial, time, u_true = data
            preds = self(spatial, time)

            # if u_true is a single Tensor, wrap it into a dict
            if not isinstance(u_true, dict):
                u_true = {self.model.output_names[0]: u_true}

            # only compute losses on the intersection of preds and u_true
            for name in set(preds.keys()).intersection(u_true.keys()):
                loss = loss + self.loss_fn(preds[name], u_true[name])

            # optional: warn if there were unmatched keys
            extra = set(u_true.keys()) - set(preds.keys())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_batch_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_batch_loss(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        spatial, time, _ = batch

        preds = self.forward(spatial, time)
        return preds

    def configure_optimizers(self):
        opt = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler_class:
            sched = self.hparams.scheduler_class(opt)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": opt}

load_from_checkpoint = True
run_pinn = False

dnn_ckpt = '20000_epochs_dnn.ckpt'
pinn_ckpt = '20000_epochs_run.ckpt'

if run_pinn:
    checkpoint = pinn_ckpt
else:
    checkpoint = dnn_ckpt

def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstorch.utils.load_data(root_path, "NLS.mat")
    exact = data["uu"]
    exact_u = np.real(exact) # N x T
    exact_v = np.imag(exact) # N x T
    exact_h = np.sqrt(exact_u**2 + exact_v**2) # N x T
    return {"u": exact_u, "v": exact_v, "h": exact_h}

time_domain = pinnstorch.data.TimeDomain(t_interval=[0, 1.57079633], t_points = 201)
spatial_domain = pinnstorch.data.Interval(x_interval= [-5, 4.9609375], shape = [256, 1])

mesh = pinnstorch.data.Mesh(root_dir='/home/sschott/CSCI582-Final-Project/pinns-torch/data',
                            read_data_fn=read_data_fn,
                            spatial_domain = spatial_domain,
                            time_domain = time_domain)

N0 = 50

in_c = pinnstorch.data.InitialCondition(mesh = mesh,
                                        num_sample = N0,
                                        solution = ['u', 'v'])

N_b = 50
pe_b = pinnstorch.data.PeriodicBoundaryCondition(mesh = mesh,
                                                 num_sample = N_b,
                                                 derivative_order = 1,
                                                 solution = ['u', 'v'])

N_f = 20000
me_s = pinnstorch.data.MeshSampler(mesh = mesh,
                                   num_sample = N_f,
                                   collection_points = ['f_v', 'f_u'])

val_s = pinnstorch.data.MeshSampler(mesh = mesh,
                                    solution = ['u', 'v', 'h'])

net = pinnstorch.models.FCN(layers = [2, 100, 100, 100, 100, 2],
                            output_names = ['u', 'v'],
                            lb=mesh.lb,
                            ub=mesh.ub)

def output_fn(outputs: Dict[str, torch.Tensor],
              x: torch.Tensor,
              t: torch.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    outputs["h"] = torch.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)

    return outputs

def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    """Define the partial differential equations (PDEs)."""
    u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
    v_x, v_t = pinnstorch.utils.gradient(outputs["v"], [x, t])

    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    v_xx = pinnstorch.utils.gradient(v_x, x)[0]

    outputs["f_u"] = u_t + 0.5 * v_xx + (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["v"]
    outputs["f_v"] = v_t - 0.5 * u_xx - (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["u"]

    return outputs

train_datasets = [me_s, in_c, pe_b]
val_dataset = val_s
datamodule = pinnstorch.data.PINNDataModule(train_datasets = [me_s, in_c, pe_b],
                                            val_dataset = val_dataset,
                                            pred_dataset = val_s)

if run_pinn:
    model = pinnstorch.models.PINNModule(net = net,
                                    pde_fn = pde_fn,
                                    output_fn = output_fn,
                                    loss_fn = 'mse')
else:
    model = FCNLightning(fcn=net)

trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=20000)

if not load_from_checkpoint:
    trainer.fit(model=model, datamodule=datamodule)
else:
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=f'/home/sschott/CSCI582-Final-Project/pinns-torch/project_testing/{checkpoint}')

collect_data = True
if collect_data:
    # this is just so that we can run enough predicts to collect data
    # scale the number in the range below if you need it to run for longer or shorter
    for _ in range(2000):
        preds_list = trainer.predict(model=model, datamodule=datamodule)
else:
    preds_list = trainer.predict(model=model, datamodule=datamodule)