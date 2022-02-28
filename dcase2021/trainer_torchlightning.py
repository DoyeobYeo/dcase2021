import torch

import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers

from torchmetrics import MeanSquaredError as MSE

from dcase2021.architectures import AutoEncoder
from dcase2021.libs import yaml_load
from dcase2021.libs import select_dirs
from dcase2021.data_loader import make_dataloader


class AutoEncoder_tl(pl.LightningModule):
    def __init__(self, input_dim=640, lr=0.001):
        super().__init__()
        self.net = AutoEncoder(input_dim=input_dim)
        self.lr = lr

        self.train_loss = MSE()
        self.val_loss = MSE()
        self.test_loss = MSE()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[50, 75],
        #     gamma=0.1
        # )
        # return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat = self.net(x)
        loss = self.train_loss(x, x_hat)

        self.log_dict(
            {'train_loss': loss},
            on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        x_hat = self.net(x)
        loss = self.val_loss(x, x_hat)

        self.log_dict(
            {'val_loss': loss},
            on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step_end(self, outputs):
        return outputs


if __name__ == '__main__':
    # "development": mode == True
    # "evaluation": mode == False
    _mode = True
    _param = yaml_load("../baseline.yaml")
    _dirs = select_dirs(param=_param, mode=_mode)
    _epoch = _param["fit"]["epochs"]
    _batch_size = _param["fit"]["batch_size"]
    _lr = _param["fit"]["lr"]

    # loop of the base directory
    for idx, target_dir in enumerate(_dirs):
        # make datasets and dataloaders for each problem
        train_dataloader = make_dataloader(
            target_dir=target_dir,
            param=_param,
            mode=_mode,
            batch_size=_batch_size
        )

        network = AutoEncoder_tl(
            input_dim=640,
            lr=_lr
        )

        trainer = pl.Trainer(
            # auto_scale_batch_size='power',
            gpus="0",
            # accelerator='ddp',
            # deterministic=True,
            max_epochs=_epoch,
            # precision=16
        )

        trainer.fit(network, train_dataloader)