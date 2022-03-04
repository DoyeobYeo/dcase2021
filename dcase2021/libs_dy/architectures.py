import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder1 = nn.Linear(input_dim, 128)
        self.bt_en1 = nn.BatchNorm1d(128)
        self.activation_en1 = nn.ReLU()

        self.encoder2 = nn.Linear(128, 128)
        self.bt_en2 = nn.BatchNorm1d(128)
        self.activation_en2 = nn.ReLU()

        self.encoder3 = nn.Linear(128, 128)
        self.bt_en3 = nn.BatchNorm1d(128)
        self.activation_en3 = nn.ReLU()

        self.encoder4 = nn.Linear(128, 128)
        self.bt_en4 = nn.BatchNorm1d(128)
        self.activation_en4 = nn.ReLU()

        self.encoder5 = nn.Linear(128, 8)
        self.bt_en5 = nn.BatchNorm1d(8)
        self.activation_en5 = nn.ReLU()

        self.decoder1 = nn.Linear(8, 128)
        self.bt_de1 = nn.BatchNorm1d(128)
        self.activation_de1 = nn.ReLU()

        self.decoder2 = nn.Linear(128, 128)
        self.bt_de2 = nn.BatchNorm1d(128)
        self.activation_de2 = nn.ReLU()

        self.decoder3 = nn.Linear(128, 128)
        self.bt_de3 = nn.BatchNorm1d(128)
        self.activation_de3 = nn.ReLU()

        self.decoder4 = nn.Linear(128, 128)
        self.bt_de4 = nn.BatchNorm1d(128)
        self.activation_de4 = nn.ReLU()

        self.decoder5 = nn.Linear(128, input_dim)
        # self.bt_de5 = nn.BatchNorm1d(input_dim)
        # self.activation_de5 = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        en_x = self.encoder1(x)
        en_x = self.bt_en1(en_x)
        en_x = self.activation_en1(en_x)

        en_x = self.encoder2(en_x)
        en_x = self.bt_en2(en_x)
        en_x = self.activation_en2(en_x)

        en_x = self.encoder3(en_x)
        en_x = self.bt_en3(en_x)
        en_x = self.activation_en3(en_x)

        en_x = self.encoder4(en_x)
        en_x = self.bt_en4(en_x)
        en_x = self.activation_en4(en_x)

        en_x = self.encoder5(en_x)
        en_x = self.bt_en5(en_x)
        encoded = self.activation_en5(en_x)

        de_x = self.decoder1(encoded)
        de_x = self.bt_de1(de_x)
        de_x = self.activation_de1(de_x)

        de_x = self.decoder2(de_x)
        de_x = self.bt_de2(de_x)
        de_x = self.activation_de2(de_x)

        de_x = self.decoder3(de_x)
        de_x = self.bt_de3(de_x)
        de_x = self.activation_de3(de_x)

        de_x = self.decoder4(de_x)
        de_x = self.bt_de4(de_x)
        de_x = self.activation_de4(de_x)

        out = self.decoder5(de_x)
        # de_x = self.bt_de5(de_x)
        # out = self.activation_de5(de_x)

        return out
