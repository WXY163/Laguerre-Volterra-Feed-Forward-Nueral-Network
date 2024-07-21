import numpy as np
import feed_data as fd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy.fft as fft
import kernels as ks
memory = 30


def load_kernel():
    h0kernel = os.getcwd()+'/kernels/h0.csv'
    h1kernel = os.getcwd()+'/kernels/h1.csv'
    h2kernel = os.getcwd()+'/kernels/h2.csv'
    h3kernel = os.getcwd()+'/kernels/h3.csv'
    h0 = np.loadtxt(h0kernel,dtype = np.float32)
    h1 = np.loadtxt(h1kernel,dtype = np.float32)
    h2 = np.loadtxt(h2kernel,dtype = np.float32)
    h3 = np.loadtxt(h3kernel,dtype = np.float32)
    h3 = h3.reshape(memory,memory,memory)
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
def calculate_h3ave(h3,memory):
    h3_ave = np.zeros((memory,memory),dtype = np.float32)
    for i in range(memory):
        h3_ave += h3[i]
    return h3_ave/memory


def conv1(h1, test_Xm,sample, output_Y):
    for i in range(sample):
        output_Y[i] = (test_Xm[i]*h1).sum()
     
def conv2(h2, test_Xm,sample, output_Y,memory):
    for sam in range(sample):
        out = 0.0
        for i in range(memory):
            for j in range(memory):
                out += test_Xm[sam, i]*test_Xm[sam,j]*h2[i,j]
        output_Y[sam] = output_Y[sam]+out        



def conv3(h3, test_Xm,sample, output_Y,memory):
    for sam in range(sample):
        out = 0.0
        for i in range(memory):
            for j in range(memory):
                for k in range(memory):
                    out += test_Xm[sam, i]*test_Xm[sam,j]*test_Xm[sam,k]*h3[i,j,k]
        output_Y[sam] = output_Y[sam]+out 

def plot_output(Y_label, Y_kernel):
    plt.figure()
    plt.plot(Y_label,'b-')
    plt.plot(Y_kernel,'r--')

def plot_fft(data):
    size = data.size
    frate = 1e2
    data = data.reshape(size)
    window = np.hamming(size)
    fftdata = data
    F = fft.fft(fftdata,size)
    freq = np.linspace(-0.5,0.5,len(F))*frate 
    freq = freq[int(size/2):] 
    freq = freq[:180]
    mag = np.abs(fft.fftshift(F))/size
    mag = mag[int(size/2):]
    mag = mag[:180]
    plt.figure()
    plt.plot( freq, mag)
    plt.xlabel("Freq(GHz)")
    plt.xticks(np.arange(0,2,step = 0.2))
def plot_data():
    trainerr_path = os.getcwd()+'/log/traerr.csv'
    valerr_path = os.getcwd()+'/log/valerr.csv'
    modeloutputpath = os.getcwd()+'/log/modeloutput.csv'
    modeloutput = np.loadtxt(modeloutputpath, dtype = np.float32)
    valerr = np.loadtxt(valerr_path, np.float32)
    plt.plot(valerr)
    #plt.plot(modeloutput)

#########################################################################
h0,h1,h2,h3 = load_kernel()

print(h0)
plot_h1(h1)
plot_h2(h2)
h3ave = calculate_h3ave(h3,memory)
#np.savetxt('kernels/h3ave.csv',h3ave,delimiter = ',')
#plot_h2(h2ana)
plot_h2(h3ave)

plt.show()

