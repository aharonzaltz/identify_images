import torch
import numpy as np
import torch.nn as nn

from config import numbers_for_layer_model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.layer1 = ...
        self.nn_layers = nn.ModuleList()
        self.layers = []
        for index, next_index in enumerate(range(1, len(numbers_for_layer_model))):
            first_number = numbers_for_layer_model[index]
            next_number = numbers_for_layer_model[next_index]
            print (first_number, next_number)

            self.nn_layers.append(nn.Linear(first_number, next_number))
            self.layers.append(nn.Linear(first_number, next_number))
        self.activation = nn.ReLU()

    def forward(self, x):

        for index, layer in enumerate(self.layers):
            out = layer(x) if(index == 0) else layer(out)
            out = out if (index == len(self.layers) - 1) else self.activation(out)
        return out
