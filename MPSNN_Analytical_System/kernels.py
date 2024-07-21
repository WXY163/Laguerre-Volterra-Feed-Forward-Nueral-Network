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
""" generate analytical kernels"""
def akernel(memory):
    a = 2.0
    m = 0.5
    k = 0.1
    """first order kernel """
    h1 = np.zeros(memory, np.float32)
    h2 = np.zeros((memory,memory),np.float32)
    for i in range(memory):
        h1[i] = a/m*np.exp(-1*k*i)*np.sin(m*i)
    """second order kernel"""
    for i in range(memory):
        for j in range(memory):
            h2[i,j] = a*a/(m*m)*np.exp(-1*k*(i+j))*np.sin(m*i)*np.sin(m*j)
    return h1, h2 


def calc_kernel(order=1, memory=50, modelpath = os.getcwd()+'/model/modeltrained.pt'):
    """ model parameters"""
    input_size = memory
    output_size = 1
    hidden_size = [memory]
    activator  = [nn.Tanh()]
    model = mlp.Net(input_size, hidden_size, output_size, activator)
    model.load_state_dict(torch.load(modelpath))

    pathh1 = os.getcwd() +'/kernels/h1.csv'
    pathh2 = os.getcwd() +'/kernels/h2.csv'
    pathh3 = os.getcwd() +'/kernels/h3.csv'

    h1 = np.zeros(memory,dtype = np.float32)
    h2 = np.zeros((memory,memory),dtype = np.float32)
    h3 = np.zeros((memory,memory,memory),dtype = np.float32)
    """ read in weights """
    w = (list(model.parameters())[0]).data.numpy()
    b = (list(model.parameters())[1]).data.numpy()
    c = (list(model.parameters())[2]).data.numpy().flatten()
    


#tanh(b)
    tanhb = np.tanh(b)
#tanh(b)^2
    tanhb2= np.power(tanhb,2)
#first derivative 
    dev1_tanhb = 1-tanhb2
#second derivative
    dev2_tanhb = 2*tanhb*(tanhb2 - 1)
#third derivative 
    dev3_tanhb = 4*tanhb2*(1-tanhb2) -  2*(1-tanhb2)*(1 -tanhb2) 
    a1 = dev1_tanhb   
    a2 =dev2_tanhb
    a3 =1/6*dev3_tanhb 

    if order == 2:
        for i in range(memory):
            h1[i] = (c*a1*w[:,i]).sum()
        for i in range(memory):
            for j in range(memory):
                h2[i,j] = (c*a2*w[:,i]*w[:,j]).sum()
        np.savetxt(pathh1,h1, delimiter = ' ')
        np.savetxt(pathh2,h2, delimiter = ' ')

        return h1, h2
    elif order ==3:
        for i in range(memory):
            h1[i] = (c*a1*w[:,i]).sum()
        for i in range(memory):
            for j in range(memory):
                h2[i,j] = (c*a2*w[:,i]*w[:,j]).sum()

        for i in range(memory):
            for j in range(memory):
                for z in range(memory):
                    h3[i,j,z] = (c*a3*w[:,i]*w[:,j]*w[:,z]).sum()
        np.savetxt(pathh1,h1, delimiter = ' ')
        np.savetxt(pathh2,h2, delimiter = ' ')
        np.savetxt(pathh3,h3, delimiter = ' ')
        return h1, h2, h3
    else:
        for i in range(memory):
            h1[i] = (c*a1*w[:,i]).sum()
        np.savetxt(pathh1,h1, delimiter = ' ')
        return h1        


def kernel_plot(memory=50, order=1, kernel=np.zeros(50)):
    if order == 1:
        plt.plot(kernel)
    if order == 2:
        t1 = np.linspace(0,memory,memory,False)
        t2 = np.linspace(0,memory,memory,False)
        t1,t2 = np.meshgrid(t2,t1)
 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
      #  ax.set_zlim3d(-1e-3, 1e-3)
        surf =ax.plot_surface(t1, t2, kernel, cmap = cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf,shrink=0.5, aspect=5)
        ax.set_zticklabels([])
    if order == 3:
        t1 = np.linspace(0,mory,memory,False)
        t2 = np.linspace(0,memory,memory,False)
        t1,t2 = np.meshgrid(t2,t1)
        kernel_ave = np.zeros((memory, memory), dtype = np.float32)   
        for k in kernel:
            kernel_ave += k
        kernel_ave = kernel_ave/memory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
       # ax.set_zlim3d(-1e-3, 1e-3)
        surf =ax.plot_surface(t1, t2, kernel_ave, cmap = cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf,shrink=0.5, aspect=5)



def conv(kernel, data, order,emory):
    output = np.zeros(data.shape[0], dtype = np.float32)
    for i, d in enumerate(data):
        output[i] = (kernel*d).sum()
    if order == 2:
        for s, d in enumerate(data):
            for i in range(memory):
                for j in range(memory):
                    output[s] += d[i]*d[j]*kernel[i,j]
    if order == 3:
        for s, d in enumerate(data):
            for i in range(memory):
                for j in range(memory):
                    for k in range(memory):
                        output[s] += d[i]*d[j]*d[k]*kernel[i,j,k]
    return output
        




