import torch.nn as nn
import torch
import torch.nn.functional as tf
import mlp
import numpy as np
import feed_data as fd
import torch.utils.data as utils
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import mlp_plot
import os
import sys


tranciver = os.getcwd()+'/data/Wave.txt'
path_Y = os.getcwd() + '/data/yt_train.txt'
path_vj = os.getcwd() + '/data/vj.txt'

modelpath = os.getcwd()+'/model/j10m150a90pam4new.pt'
trainerrpath = os.getcwd()+'/log/traerr.csv'
validateerrpath = os.getcwd()+'/log/valerr.csv'

load_tcdata = fd.tranciver_load(tranciver)

memory =150
J =10 
alpha = 0.9
input_size = J
output_size = 1
hidden_size = [J]
activator  = [mlp.SVN()]
batch_size = 50
model = mlp.Net(input_size, hidden_size, output_size, activator)
model.load_state_dict(torch.load(modelpath))
x = np.loadtxt(tranciver)
#xin = x[300:]
Vj = load_tcdata.calculate_vj_conv_xonly(x,alpha,J,memory)  
#Vjr = load_tcdata.calculate_vj(xin,alpha,J) 
#Vj0 = load_tcdata.calculate_vj0(x, alpha,J)


np.savetxt(path_vj, Vj, delimiter =' ');
#data_Xm = np.loadtxt(path_vj)
sys.exit()



tensor_x = torch.stack([torch.Tensor(i) for i in Vj])


output_Y = np.zeros(x.size)
for i, x_in in enumerate(tensor_x):
    output = model(x_in)
    output_Y[i] = output.data.numpy().flatten()

plt.plot(output_Y[1000:1500])
plt.show()


