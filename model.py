import torch
import numpy as np
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.layer1 = ...
        self.layer1 = nn.Linear(4761, 400)
        self.layer2 = nn.Linear(400, 100)
        # self.layer3 = nn.Linear(400,100)
        self.layer3 = nn.Linear(100, 10)
        self.activation1 = nn.ReLU()
        # self.activation2 = nn.Tanh()

    def forward(self, x):
        out = self.layer1(x)
        out = self.activation1(out)
        out = self.layer2(out)
        out = self.activation1(out)
        out = self.layer3(out)
        # out = self.activation1(out)
        # out = self.layer4(out)

        return out