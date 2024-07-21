import numpy as np
import os
import sys
from matplotlib import pyplot as plt
class gaussian_data:
    def __init__(self, save=False, directory = os.getcwd(), data_type='train'):
        if(save == True):
            if(os.path.isdir(directory)):
                self.dir = directory + '/data'
            else:
                sys.exit("invalid path")

        self.save = save
        self.data_type = data_type

    def response(self,x):
        a = 2
        m = 0.5
        k = 0.1
        time_series = np.linspace(0,x,x, False)
        res = a/m*np.exp((-1)*k*time_series)*np.sin(m*time_series)
        return res

    def xt(self,samples,memory):
        d = np.random.normal(0.0,1.0,samples)
        x = np.zeros((samples -memory, memory),dtype = np.float32)
        for i in range(samples - memory):
            x[i,:] = d[i:i + memory]
        if(self.save == True):
            if(self.data_type == 'train'):
                path = self.dir + '/xt_train.csv'
            if(self.data_type == 'validate'):               
                path = self.dir + '/xt_validate.csv'
            if(self.data_type == 'test'):               
                path = self.dir + '/xt_test.csv'
            np.savetxt(path,d,delimiter=",")

        return d,x 

    def yt(self,xt,memory):
        sample = xt.shape[0]
        res = self.response(memory) 
        ta = np.zeros(memory)

        y = np.zeros(sample - memory,dtype=np.float32)
        for i in range(sample - memory):
            ta = xt[i:i+memory]
            tb = res[::-1]
            y[i] = (ta*tb).sum()
        # second order system
        #y = np.power(y,2).reshape(sample - memory,1)
        #first ordersystem
        y = y.reshape(sample - memory,1)

        if(self.save == True):
            if(self.data_type == 'train'):
                path = self.dir + '/yt_train.csv'
            if(self.data_type == 'validate'):               
                path = self.dir + '/yt_validate.csv'
            if(self.data_type == 'test'):               
                path = self.dir + '/yt_test.csv'
            np.savetxt(path,y,delimiter=",")
        return y 
class data_load:
    def __init__(self, pathx, pathy):
        if not os.path.exists(pathx):
            print(pathx+" is not valid!\n")
        self.pathx = pathx
        if not os.path.exists(pathy):
            print(pathy+" is not valid!\n")
        self.pathy = pathy

    def loadx(self, memory):
        xt = np.genfromtxt(self.pathx,delimiter=',', dtype=np.float32)
        if(xt.ndim >1):
            sys.exit("xdata is in wrong dimension\n")
            
        x = np.zeros((xt.size -memory, memory),dtype = np.float32)
        for i in range(xt.size - memory):
            x[i,:] = xt[i:i + memory]
        return x
    def loady(self):
        yt = np.genfromtxt(self.pathy,delimiter=',', dtype=np.float32)
        return yt.reshape(yt.size,1)

class tranciver_load:
    def __init__(self,path):
        if not os.path.exists(path):
            sys.exit(path + " is not valide!")
        self.path = path
    def data_read(self,memory):
        data = np.loadtxt(self.path)
        y = data[:,3]
        label = y[memory:]
        input_x = data[:,1]
        x = np.zeros((input_x.size - memory, memory), dtype = np.float32)
        for i in range(input_x.size - memory):
            x[i,:] = input_x[i:i + memory]
            
        return label.reshape(label.size,1), x 
