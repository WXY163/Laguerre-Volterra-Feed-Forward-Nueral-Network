import numpy as np
import feed_data as fd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import kernels as ks
memory = 50


tranciver = os.getcwd()+'/data/data.csv'
modelpath = os.getcwd()+'/model/modeltrained.pt'

h1ana, h2ana = ks.akernel(memory)

h1nn,h2nn = ks.calc_kernel( 2, 50, modelpath) 
h1nn = h1nn[::-1]
h2nn = h2nn[::-1, ::-1]
#ks.kernel_plot(memory, 1, h1nn[::-1])
#ks.kernel_plot(memory, 2, h2nn[::-1,::-1])
#ks.kernel_plot(memory, 1,h1ana)
#ks.kernel_plot(memory,2,h2ana)
h1kernel = os.getcwd()+'/kernels/h1.csv'
h2kernel = os.getcwd()+'/kernels/h2.csv'
h3kernel = os.getcwd()+'/kernels/h3.csv'
test_sample = 1000
load_tcdata = fd.tranciver_load(tranciver)
h1 = np.loadtxt(h1kernel,dtype = np.float32)
h2 = np.loadtxt(h2kernel,dtype = np.float32)
h3 = np.loadtxt(h3kernel,dtype = np.float32)
#h3 = h3.reshape(memory,memory,memory)

data_Y, data_Xm = load_tcdata.data_read(memory)  
test_Y = data_Y[:1000]
test_Xm = data_Xm[:1000]
sample  = test_sample

output_Y = np.zeros(test_sample)
output_Ytruc = np.zeros(test_sample)
"""
plt.plot(h1ana,'b-', label='kernels from equation')
plt.plot(h1nn,'r--', label = 'kernels from nn')
plt.xlabel('time')
plt.axhline(y=0, color = 'k', linewidth = 0.5)
plt.legend()
#plt.ylim(-1, 1)
plt.xlim(0,50)
plt.show()
"""
t1 = np.linspace(0,memory,memory,False)
t2 = np.linspace(0,memory,memory,False)
t1,t2 = np.meshgrid(t2,t1)
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
      #  ax.set_zlim3d(-1e-3, 1e-3)
surf = ax.plot_surface(t1, t2, np.zeros((50,50)), cmap = cm.coolwarm, linewidth=1, antialiased=False, alpha=0.75)
ax.scatter(t1,t2,h2nn, c='b', marker='o', s= 0.5, label = 'kernels from nn')
fig.colorbar(surf,shrink=0.5, aspect=5)
#ax.set_zticklabels([])
ax.set_zlim3d(-0.1,0.1)
ax.set_xlabel('time')
ax.set_ylabel('time')
plt.show()
