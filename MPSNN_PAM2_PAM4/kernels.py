import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import torch.nn as nn
import torch
import mlp
import os
import feed_data as fd
import torch.utils.data as utils
import math
# specify memory length
memory  = 300
def calc_kernel(order):
    """ model parameters"""
    input_size = memory
    output_size = 1
    hidden_size = [memory]
    activator  = [nn.Tanh()]
    model = mlp.Net(input_size, hidden_size, output_size, activator)
    modelpath = os.getcwd()+'/model/modeltrained.pt'
    """ """
    model.load_state_dict(torch.load(modelpath))
    h1 = np.zeros(memory,dtype = np.float64)
    h2 = np.zeros((memory,memory),dtype = np.float64)
    h3 = np.zeros((memory,memory,memory),dtype = np.float64)
    """ read in weights """
    w = (list(model.parameters())[0]).data.numpy()
    b = (list(model.parameters())[1]).data.numpy()
    c = (list(model.parameters())[2]).data.numpy().flatten()
    d = (list(model.parameters())[3]).data.numpy()
   #calculate h0
    h0 = d
    h0 += (c*np.power(b,order)).sum() 
   #calculate h1
    for i in range(memory):
        h1[i] = 3*(c*np.power(b,(order-1))*w[:,i]).sum()
    #calculate h2
    for i in range(memory):
        for j in range(memory):
            h2[i,j] = 3*(c*np.power(b,order-2)*w[:,i]*w[:,j]).sum()
   #calcualte h3 
    for i in range(memory):
        for j in range(memory):
            for z in range(memory):
                h3[i,j,z] = (c*w[:,i]*w[:,j]*w[:,z]).sum()

    return h0, h1, h2, h3

  #save them 
pathh0 = os.getcwd() +'/kernels/h0.csv'
pathh1 = os.getcwd() +'/kernels/h1.csv'
pathh2 = os.getcwd() +'/kernels/h2.csv'
pathh3 = os.getcwd() +'/kernels/h3.csv'
h0,h1,h2,h3 = calc_kernel(3)
h3 = h3.reshape(memory,memory*memory)
np.savetxt(pathh0,h0, delimiter = ' ')
np.savetxt(pathh1,h1, delimiter = ' ')
np.savetxt(pathh2,h2, delimiter = ' ')
np.savetxt(pathh3,h3, delimiter = ' ')
