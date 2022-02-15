import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from dcase2021.architectures import AutoEncoder
from dcase2021.libs import yaml_load
from dcase2021.libs import select_dirs
from dcase2021.data_loader import make_dataloader


device = 'cuda'


def train_one_epoch(net, train_dataloader, num_epoch):
    net.train()

    for inputs in tqdm(train_dataloader, desc="Epoch: "+str(num_epoch)):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    net = AutoEncoder(input_dim=640)
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # "development": mode == True
    # "evaluation": mode == False
    _mode = True
    _param = yaml_load("../baseline.yaml")
    _dirs = select_dirs(param=_param, mode=_mode)
    _epoch = _param["fit"]["epochs"]
    _batch_size = _param["fit"]["batch_size"]

    # loop of the base directory
    for idx, target_dir in enumerate(_dirs):
        # make datasets and dataloaders for each problem
        train_dataloader = make_dataloader(
            target_dir=target_dir,
            param=_param,
            mode=_mode,
            batch_size=_batch_size
        )

        for epoch in range(1, _epoch + 1):
            train_one_epoch(net, train_dataloader, num_epoch=epoch)





