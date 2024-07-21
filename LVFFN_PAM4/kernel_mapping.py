import numpy as np
import feed_data as fd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy.fft as fft
import kernels as ks
import laugerre as lgr
import sys
memory = 150
R = 10
alpha = 0.9
def load_theta(mem):
    h0kernel = os.getcwd()+'/kernels/theta0.csv'
    h1kernel = os.getcwd()+'/kernels/theta1.csv'
    h2kernel = os.getcwd()+'/kernels/theta2.csv'
    h3kernel = os.getcwd()+'/kernels/theta3.csv'
    theta0 = np.loadtxt(h0kernel,dtype = np.float32)
    theta1 = np.loadtxt(h1kernel,dtype = np.float32)
    theta2 = np.loadtxt(h2kernel,dtype = np.float32)
    theta3 = np.loadtxt(h3kernel,dtype = np.float32)
    theta3 = theta3.reshape(mem,mem,mem)
    return theta0,theta1, theta2, theta3
def plot_h1(h1):
    plt.plot(h1)
def plot_h2(h2,mem):
    t1 = np.linspace(0,mem,mem,False)
    t2 = np.linspace(0,mem,mem,False)
    t1,t2 = np.meshgrid(t1,t2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf =ax.plot_surface(t1, t2,h2 , cmap = cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    
    ax.set_zticklabels([])
    ax.set_xlabel('Time unit')
    ax.set_ylabel('Time unit')
    #ax.view_init(azim = 135)
    #ax.set_zlim3d(-1e-5,1e-5)
    #fig.savefig('h2_analytical.png', format = 'png', dpi = 600)
def calculate_h3ave(h3,mem):
    h3_ave = np.zeros((mem,mem),dtype = np.float32)
    for i in range(mem):
        h3_ave += h3[i]
    return h3_ave/mem

def kernel_mapping(alpha,R,mem):
    h0,h1,h2,h3 = load_theta(R)
    if h1.size != R:
        sys.exit("number of functions is not right!")
    k0 = np.array([h0])
    k1 = np.zeros(mem) 
    b = lgr.calculate_b(alpha,R,mem)
    for m in range(mem):
        k1[m] += (h1*b[:,m]).sum()  
    k2 = np.zeros((mem,mem),np.float32) 
    for i in range(mem):
        for j in range(mem):
            temp = np.matmul(b[:,i].reshape(R,1),b[:,j].reshape(1,R)).reshape(R,R)
            k2[i,j] = (h2*temp).sum()
    k3 = np.zeros((mem,mem,mem),np.float32)
    temp3 = np.zeros((R,R,R), np.float32)
    for i in range(mem):
        print(i)
        for j in range(mem):
            for k in range(mem):
                temp3[:] = np.matmul(b[:,j].reshape(R,1),b[:,k].reshape(1,R)).reshape(R,R)
                temp3 = temp3*b[:,i][:,np.newaxis,np.newaxis]
                k3[i,j,k] = (h3*temp3).sum()
    return k0,k1,k2,k3

#########################################################################
k0path = os.getcwd()+'/kernels/h0.csv'
k1path = os.getcwd()+'/kernels/h1.csv'
k2path = os.getcwd()+'/kernels/h2.csv'
k3path = os.getcwd()+'/kernels/h3.csv'

#h0,h1,h2,h3 = load_theta(R)
h0,h1,h2,h3 = kernel_mapping(alpha,R,memory) 
np.savetxt(k0path,h0, delimiter = ' ')
np.savetxt(k1path,h1, delimiter = ' ')
np.savetxt(k2path,h2, delimiter = ' ')
np.savetxt(k3path,h3.reshape(memory,memory*memory), delimiter = ' ')

