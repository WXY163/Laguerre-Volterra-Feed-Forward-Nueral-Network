import numpy as np
import torch.nn as nn

import torch
import mlp
import os
import matplotlib.pyplot as plt
import feed_data as fd
import torch.utils.data as utils

memory = 50
input_size = memory
output_size = 1
hidden_size = [memory]
activator  = [nn.Tanh()]
model = mlp.Net(input_size, hidden_size, output_size, activator)
modelpath = os.getcwd()+'/model/modeltrained_save50.pt'
model.load_state_dict(torch.load(modelpath))
fig = plt.figure()

w = (list(model.parameters())[0]).data.numpy()
b = (list(model.parameters())[1]).data.numpy()
c = (list(model.parameters())[2]).data.numpy().flatten()

print(b)
