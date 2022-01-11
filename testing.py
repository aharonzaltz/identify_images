import torch
from random import randrange

from model import Net

arr = []
for i in range(4):
    arr.append(float(randrange(10)))

vector = torch.tensor(arr)

print (vector)

net = Net()
print (net.layers[0].weight, net.layers[0].bias)
out = net.layers[0](vector)
out = net.activation(out)
out = net.layers[1](out)
out = net.activation(out)
out = net.layers[2](out)
out1 = net(vector)
print (out, out1)
