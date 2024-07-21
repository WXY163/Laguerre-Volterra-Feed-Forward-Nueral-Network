import numpy as np
import feed_data as fd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy.fft as fft
import kernels as ks
import laugerre as lgr
memory = 300
R =25 
alpha = 0.83
def load_kernel(mem):
    h0kernel = os.getcwd()+'/kernels/h0.csv'
    h1kernel = os.getcwd()+'/kernels/h1.csv'
    h2kernel = os.getcwd()+'/kernels/h2.csv'
    h3kernel = os.getcwd()+'/kernels/h3.csv'
    h0 = np.loadtxt(h0kernel,dtype = np.float64)
    h1 = np.loadtxt(h1kernel,dtype = np.float64)
    h2 = np.loadtxt(h2kernel,dtype = np.float64)
    h3 = np.loadtxt(h3kernel,dtype = np.float64)
    h3 = h3.reshape(mem,mem,mem)
    return h0,h1, h2, h3
def plot_h1(h1):
    plt.plot(h1)
def plot_h2(h2):
    t1 = np.linspace(0,memory,memory,False)
    t2 = np.linspace(0,memory,memory,False)
    t1,t2 = np.meshgrid(t1,t2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf =ax.plot_surface(t1, t2,h2[::-1,::-1] , cmap = cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    
    ax.set_zticklabels([])
    ax.set_xlabel('Time unit')
    ax.set_ylabel('Time unit')
    #ax.view_init(azim = 135)
    #ax.set_zlim3d(-1e-5,1e-5)
    fig.savefig('h2_analytical.png', format = 'png', dpi = 600)
def calculate_h3ave(h3,mem):
    h3_ave = np.zeros((mem,mem),dtype = np.float64)
    for i in range(mem):
        h3_ave += h3[i]
    return h3_ave/mem

def kernel_mapping(alpha,R,mem):
    h0,h1,h2,h3 = load_kernel(R)

    if h1.size != R:
        sys.exit("number of functions is not right!")
    k0 = h0 
    k1 = np.zeros(mem) 
    for m in range(mem):
        for l in range(R):
            k1[m] += h1[l]*lgr.calculate_bjm(alpha,l,m)  
    """
    k2 = np.zeros((mem,mem),np.float32) 
    for i in range(mem):
        for j in range(mem):
            for l1 in range(R):
                for l2 in range(R):
                    k2[i,j] += h2[l1,l2]*lgr.calculate_bjm(alpha,l1,i)*lgr.calculate_bjm(alpha,l2,j)
    """
    return k0,k1

#########################################################################
#h0,h1,h2,h3 = load_kernel()
h0,h1 = kernel_mapping(alpha,R,memory) 
#h1ana, h2ana = ks.akernel(memory)

print(h0)
plot_h1(h1)
#plot_h2(h2)
#h3ave = calculate_h3ave(h3,memory)
#np.savetxt('kernels/h3ave.csv',h3ave,delimiter = ',')
#plot_h2(h2ana)
#plot_h2(h3ave)

plt.show()

