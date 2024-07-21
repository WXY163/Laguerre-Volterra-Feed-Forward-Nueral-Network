import numpy as np
import feed_data as fd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy.fft as fft
import kernels as ks
import feed_data as fd
memory =40 
R = 20
tranciver = os.getcwd()+'/data/255.txt'
load_tcdata = fd.tranciver_load(tranciver)
Y_label, Xm = load_tcdata.data_read(memory)
Y_label = Y_label.reshape(Y_label.size)
def load_kernel(mem):
    h0kernel = os.getcwd()+'/kernels/h0.csv'
    h1kernel = os.getcwd()+'/kernels/h1.csv'
    h2kernel = os.getcwd()+'/kernels/h2.csv'
    h3kernel = os.getcwd()+'/kernels/h3.csv'
    h0 = np.loadtxt(h0kernel,dtype = np.float32)
    h1 = np.loadtxt(h1kernel,dtype = np.float32)
    h2 = np.loadtxt(h2kernel,dtype = np.float32)
    h3 = np.loadtxt(h3kernel,dtype = np.float32)
    h3 = h3.reshape(mem,mem,mem)
    return h0,h1, h2, h3
def load_theta(R):
    theta0 = os.getcwd()+'/kernels/theta0.csv'
    theta1 = os.getcwd()+'/kernels/theta1.csv'
    theta2 = os.getcwd()+'/kernels/theta2.csv'
    theta3 = os.getcwd()+'/kernels/theta3.csv'
    h0 = np.loadtxt(theta0,dtype = np.float32)
    h1 = np.loadtxt(theta1,dtype = np.float32)
    h2 = np.loadtxt(theta2,dtype = np.float32)
    h3 = np.loadtxt(theta3,dtype = np.float32)
    h3 = h3.reshape(R,R,R)
    return h0,h1, h2,h3 
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
    fig.savefig('h2_analytical.png', format = 'png', dpi = 600)
def calculate_h3ave(h3,memory):
    h3_ave = np.zeros((memory,memory),dtype = np.float32)
    for i in range(memory):
        h3_ave += h3[i]
    return h3_ave/memory


def conv1(h1, test_Xm,sample):
    output_Y = np.zeros(sample, np.float32)
    for i in range(sample):
        output_Y[i] = (test_Xm[i]*h1).sum()
    return output_Y
     
def conv2(h2, test_Xm,sample,memory):
    output_Y = np.zeros(sample,np.float32)
    for sam in range(sample):
        output_Y[sam] = (h2*np.matmul(test_Xm[sam].reshape(memory,1), test_Xm[sam].reshape(1,memory))).sum()        
    return output_Y



def conv3(h3, test_Xm,sample, memory):
    temp = np.zeros((memory,memory,memory),np.float32)
    output_Y = np.zeros(sample,np.float32)
    for sam in range(sample):
        temp[:] = np.matmul(test_Xm[sam].reshape(memory,1), test_Xm[sam].reshape(1,memory)) 
        temp = temp*test_Xm[sam][:,np.newaxis, np.newaxis]
        output_Y[sam] = (h3*temp).sum() 
    return output_Y

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
h0,h1,h2,h3 = load_kernel(memory)
#h0,h1,h2,h3 = load_theta(R)
print(h0)
plot_h1(h1)
plot_h2(h2,memory)
h3ave = calculate_h3ave(h3,memory)
#np.savetxt('kernels/h3ave.csv',h3ave,delimiter = ',')
#plot_h2(h2ana)
plot_h2(h3ave,memory)
"""
Xm = Xm[:,::-1]
output_Y = h0 + conv1(h1,Xm,80)+conv2(h2,Xm,80,memory)+conv3(h3,Xm,80,memory)
#output_Y = conv2(h2,Xm,1000,memory)
plt.plot(Y_label)
plt.plot(output_Y)
"""
plt.show()
