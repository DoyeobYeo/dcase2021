import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from dcase2021.architectures import AutoEncoder
from dcase2021.libs import yaml_load
from dcase2021.libs import select_dirs
from dcase2021.data_loader import make_dataloader


def train_one_epoch(net, train_dataloader, num_epoch, device):
    net.train()

    for inputs in tqdm(train_dataloader, desc="Epoch: "+str(num_epoch)):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    _base_dir = '/media/yeody/DATA_2TB/dcase2021'

    _use_cuda = torch.cuda.is_available()
    _device = torch.device("cuda" if _use_cuda else "cpu")

    net = AutoEncoder(input_dim=640)
    net.to(_device)

    criterion = nn.MSELoss()

    # "development": mode == True
    # "evaluation": mode == False
    _mode = True
    _param = yaml_load(os.path.join(_base_dir, 'baseline.yaml'))
    _dirs = select_dirs(base_dir=_base_dir, param=_param, mode=_mode)

    _epoch = _param["fit"]["epochs"]
    _batch_size = _param["fit"]["batch_size"]
    _validation_split = _param["fit"]["validation_split"]
    _lr = _param["fit"]["lr"]

    # parameters for DataLoader
    _num_workers = 8
    _prefetch_factor = 4
    _pin_memory = True
    _persistent_workers = True

    optimizer = optim.Adam(net.parameters(), lr=_lr)


    # loop of the base directory
    for idx, target_dir in enumerate(_dirs):
        # make datasets and dataloaders for each problem
        train_dataloader, val_dataloader = make_dataloader(
            target_dir=target_dir,
            param=_param,
            mode=_mode,
            batch_size=_batch_size,
            validation_split=_validation_split,
            num_workers=_num_workers,
            prefetch_factor=_prefetch_factor,
            pin_memory=_pin_memory,
            persistent_workers=_persistent_workers
        )

        for epoch in range(1, _epoch + 1):
            train_one_epoch(net, train_dataloader, num_epoch=epoch, device=_device)



