import torch
import numpy as np
import torch.nn as nn

from config import numbers_for_layer_model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layersCount = 0
        for index, next_index in enumerate(range(1, len(numbers_for_layer_model))):
            first_number = numbers_for_layer_model[index]
            next_number = numbers_for_layer_model[next_index]
            print (first_number, next_number)

            self.layersCount +=1
            setattr(self, 'layer' + str(index +1), nn.Linear(first_number, next_number))
        self.activation = nn.ReLU()


    def forward(self, x):


        for index in range(2, self.layersCount):

            out = self.layer1(x) if index == 2 else self.activation(out)
            out = getattr(self, 'layer' + str(index))(out)
        return out


