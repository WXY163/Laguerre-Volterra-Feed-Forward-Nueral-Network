import numpy as np
import feed_data as fd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import kernels as ks
memory = 300
test_sample = 2000
import feed_data as fd

tranciver = os.getcwd()+'/data/Vdiff_data.txt'
modelpath = os.getcwd()+'/model/modeltrained.pt'
figurepath = os.getcwd() + '/h1_highspeedlink.png'
trainerrpath = os.getcwd()+'/log/traerr.csv'
validateerrpath = os.getcwd()+'/log/valerr.csv'


vrotate =35
hrotate = 225  

#trainerr = np.genfromtxt(trainerrpath, delimiter = ',',dtype  = np.float32)
#validerr = np.genfromtxt(validateerrpath, delimiter = ',',dtype  = np.float32)
def load_kernel():
    h0kernel = os.getcwd()+'/kernels/h0.csv'
    h1kernel = os.getcwd()+'/kernels/h1.csv'
    h2kernel = os.getcwd()+'/kernels/h2.csv'
    h3kernel = os.getcwd()+'/kernels/h3.csv'
    h0 = np.loadtxt(h0kernel,dtype = np.float64)
    h1 = np.loadtxt(h1kernel,dtype = np.float64)
    h2 = np.loadtxt(h2kernel,dtype = np.float64)
    h3 = np.loadtxt(h3kernel,dtype = np.float64)
    h3 = h3.reshape(memory,memory,memory)
    return h0,h1, h2,h3
def plot_h1(h1):
    plt.plot(h1,'k-')
    plt.xlabel('Time unit')
    plt.ylabel('Magnitude')
    #plt.axhline(y=0, color ='k', linewidth = 0.5)
    #plt.savefig('hs_h1.png',format='png',dpi = 600)
def plot_h2(h2):
    t1 = np.linspace(0,memory,memory,False)
    t2 = np.linspace(0,memory,memory,False)
    t1,t2 = np.meshgrid(t1,t2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf =ax.plot_surface(t1, t2,h2 , cmap = cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    #ax.set_zlim3d(-1e-5,1e-5)
    #ax.set_zticklabels([])
    #ax.set_zlabel('Magnitude')
    #ax.set_zlim3d(-0.1,0.1)
    ax.set_xlabel('Time unit')
    ax.set_ylabel('Time unit')
    #fig.savefig('hs_h2.png', format = 'png', dpi = 600)


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
    path = os.getcwd()+'/output_Y.txt'
    for sam in range(sample):
        out = 0.0
        for i in range(memory):
            for j in range(memory):
                for k in range(memory):
                    out += test_Xm[sam, i]*test_Xm[sam,j]*test_Xm[sam,k]*h3[i,j,k]
        output_Y[sam] = output_Y[sam]+out 
        if(sam%100 == 0):
            np.savetxt(path,output_Y)

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
def plot_err():
    trainerr_path = os.getcwd()+'/log/traerr.csv'
    valerr_path = os.getcwd()+'/log/valerr.csv'
    valerr = np.loadtxt(valerr_path, np.float32)
    trainerr = np.loadtxt(trainerr_path, np.float32)

    plt.plot(valerr,'r--', label = 'validation error')
    plt.plot(trainerr,'k-',label = 'train error')
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.legend()
    #plt.axhline(y=0, color ='k', linewidth = 0.5)
    plt.savefig('error.png',format='png',dpi = 600)


def plot_data():
    memory = 500
    modeloutputpath = os.getcwd()+'/log/output_Y.csv'
    load_tcdata = fd.tranciver_load(tranciver)
    data_Y, data_Xm = load_tcdata.data_read(memory)  
    test_Y = data_Y[1000:3000]
    test_X = data_Xm[1000:3000,0]

    modeloutput = np.loadtxt(modeloutputpath, dtype = np.float32)

    plt.plot(test_Y,'k-',label = 'measured output')
    plt.plot(modeloutput[1000:3000],'r--',label = 'reconstructed output')
    plt.xlabel('Sample Number')
    plt.ylabel('Voltage(V)')
    plt.ylim([-0.2, 0.25])
    plt.legend()
    plt.savefig('output.png',format='png',dpi = 600)

   # plt.plot(test_X)



#########################################################################
"""
path = os.getcwd()+'/log/output_Y.csv'
load_tcdata = fd.tranciver_load(tranciver)
data_Y, data_Xm = load_tcdata.data_read(memory)  
test_Y = data_Y[2000:2500]
output_Y = np.zeros(test_Y.shape)
test_Xm = data_Xm[2000:2500]
sample  = test_Y.size
"""
h3avepath = 'log/h3ave.csv'
h0,h1,h2,h3 = load_kernel()
#conv1(h1,test_Xm, sample,output_Y)
#conv2(h2,test_Xm, sample,output_Y,memory)
#conv3(h3,test_Xm, sample,output_Y,memory)
#output_Y = np.loadtxt(path,np.float32)
#output_Y += h0
#plot_output(test_Y, output_Y)

print(h0)
plot_h1(h1[::-1])
plot_h2(h2[::-1, ::-1])
h3ave = calculate_h3ave(h3,memory)
np.savetxt(h3avepath, h3ave, delimiter = ',')
plot_h2(h3ave)
#plot_fft(test_Y)

#plot_data()
plt.show()



