import torch
import numpy as np
import torch.nn as nn

from config import numbers_for_layer_model


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        self.linears = []
        for index, next_index in enumerate(range(1, len(numbers_for_layer_model))):
            first_number = numbers_for_layer_model[index]
            next_number = numbers_for_layer_model[next_index]
            # print (first_number, next_number)

            self.linears.append(nn.Linear(first_number, next_number))

        self.layers = nn.ModuleList(self.linears)
        self.activation = nn.ReLU()

    def forward(self, input):
        out = None
        for index, layer in enumerate(self.layers):
            out = layer(input if index == 0 else out)
            if not index == len(self.layers) - 1:
                out = self.activation(out)
        return out
