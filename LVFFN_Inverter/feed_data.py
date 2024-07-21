import numpy as np
import os
import sys
import operator as op
import functools

import scipy.special as scipy_special
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
    def ncr(self, n, r):
        r = min(r, n-r)
        numer = functools.reduce(op.mul, range(n, n-r, -1), 1)
        denom = functools.reduce(op.mul, range(1, r+1), 1)
        return numer//denom

    def calculate_bjm(self,a,j,m):
        elementM = np.arange(m+1)
        elementJ = np.arange(j+1)
        if m < j:
            combineMK = np.zeros(m+1)
            combineJK = np.zeros(m+1)
            for i in range(m+1):
                combineMK[i] = self.ncr(m,i)
                combineJK[i] = self.ncr(j,i)
            ceff = np.ones(m+1)* -1.0 
            ceff[::2] = 1.0
            powera = np.power(a,j - elementM)
            power1minusa = np.power((1-a),  elementM)
            rn = np.sqrt(a**(m-j))*np.sqrt(1-a)*np.sum(ceff*combineMK*combineJK*powera*power1minusa)
            return rn
        else:
            combineMK = np.zeros(j+1)
            combineJK = np.zeros(j+1)
            for i in range(j+1):
                combineMK[i] = self.ncr(m,i)
                combineJK[i] = self.ncr(j,i)
            ceff = np.ones(j+1)* -1.0 
            ceff[::2] = 1.0
            powera = np.power(a,elementJ[::-1])
            power1minusa = np.power((1-a), elementJ)
            rn = np.sqrt(a**(m-j))*np.sqrt(1-a)*np.sum(ceff*combineMK*combineJK*powera*power1minusa)
            return rn

    def calculate_b(self,alpha, J, M):
        m = M
        j = J
        b = np.zeros((j,m),np.float32)
        for i in range(j):
            for k in range(m):
                b[i,k] = self.calculate_bjm(alpha,i,k)
        return b


    
    def getM(self,alpha):
        M = (-30 - np.log(1-alpha))/np.log(alpha)
        return M

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

    def yt(self,xt,memory,order):
        sample = xt.shape[0]
        res = self.response(memory) 
        ta = np.zeros(memory)
        y = np.zeros(sample - memory,dtype=np.float32)
        for i in range(sample - memory):
            ta = xt[i:i+memory]
            tb = res[::-1]
            y[i] = (ta*tb).sum()
        if order == 2:
            y = np.power(y,2)

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
    def calculate_vj_conv(self,alpha,J,memory,samples):
        x,xm = self.xt(samples, memory)
        sample = xm.shape[0]
        bj = self.calculate_b(alpha, J, memory)
        vj = np.zeros((sample,J))
        for i, xt in enumerate(xm):
            for j, bl in  enumerate(bj):
                vj[i,j] = (xt*bl[::-1]).sum()
        if(self.save == True):
            path = self.dir + '/vj_train.csv'
            np.savetxt(path,vj,delimiter = ",")
        return vj
    def calculate_vj_recur(self,alpha, L,memory, samples):
        x,xm = self.xt(samples, memory)
        N = len(x)
        beta = np.sqrt(alpha)
        V = np.zeros((N,L))
        V[0,0] = np.sqrt(1-alpha)*x[0]

        for n in range(1,N):
            V[n,0] = beta*V[n-1,0] +np.sqrt(1-alpha)*x[n]
        for j in range(1, L):
            V[0,j] = beta*V[0,j-1]
            for k in range(1,N):
                V[k,j] = beta * (V[k-1, j] + V[k, j - 1]) - V[k-1, j-1]
        return (V)

        
class data_load:
    def __init__(self, pathx, pathy, pathvj):
        if not os.path.exists(pathx):
            print(pathx+" is not valid!\n")
        self.pathx = pathx
        if not os.path.exists(pathy):
            print(pathy+" is not valid!\n")
        self.pathy = pathy
        if not os.path.exists(pathvj):
            print(pathvj+" is not valid!\n")
        self.pathvj = pathvj
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
    def loadvj(self):
        vj = np.genfromtxt(self.pathvj,delimiter=',', dtype=np.float32)
        return vj
        return label.reshape(label.size,1), x 
    



class tranciver_load:
    def __init__(self,path):
        if not os.path.exists(path):
            sys.exit(path + " is not valide!")
        self.path = path
    def ncr(self, n, r):
        r = min(r, n-r)
        numer = functools.reduce(op.mul, range(n, n-r, -1), 1)
        denom = functools.reduce(op.mul, range(1, r+1), 1)
        return numer//denom

    def calculate_bjm(self,a,j,m):
        elementM = np.arange(m+1)
        elementJ = np.arange(j+1)
        if m < j:
            combineMK = np.zeros(m+1)
            combineJK = np.zeros(m+1)
            for i in range(m+1):
                combineMK[i] = self.ncr(m,i)
                combineJK[i] = self.ncr(j,i)
            ceff = np.ones(m+1)* -1.0 
            ceff[::2] = 1.0
            powera = np.power(a,j - elementM)
            power1minusa = np.power((1-a),  elementM)
            rn = np.sqrt(a**(m-j))*np.sqrt(1-a)*np.sum(ceff*combineMK*combineJK*powera*power1minusa)
            return rn
        else:
            combineMK = np.zeros(j+1)
            combineJK = np.zeros(j+1)
            for i in range(j+1):
                combineMK[i] = self.ncr(m,i)
                combineJK[i] = self.ncr(j,i)
            ceff = np.ones(j+1)* -1.0 
            ceff[::2] = 1.0
            powera = np.power(a,elementJ[::-1])
            power1minusa = np.power((1-a), elementJ)
            rn = np.sqrt(a**(m-j))*np.sqrt(1-a)*np.sum(ceff*combineMK*combineJK*powera*power1minusa)
            return rn

    def calculate_b(self,alpha, J, M):
        m = M
        j = J
        b = np.zeros((j,m),np.float32)
        for i in range(j):
            for k in range(m):
                b[i,k] = self.calculate_bjm(alpha,i,k)
        return b


    def data_read(self,memory):
        data = np.loadtxt(self.path, delimiter = ' ')
        y = data[:,2]
        label = y[memory:memory+80]
        input_x = data[:,1]
        x = np.zeros((80, memory), dtype = np.float32)
        for i in range(80):
            x[i,:] = input_x[i:i + memory]
        return label.reshape(label.size,1), x 
        """
        data = np.genfromtxt(self.path,delimiter=',', dtype = np.float32 )
        y  = data[:,2]
        label = y[80:] 
        #label = y[73:]
        input_x = data[80-memory:,1]
        #input_x = data[73-memory:,1]

        x = np.zeros((input_x.size - memory, memory), dtype = np.float32)
        for i in range(input_x.size - memory):
            x[i,:] = input_x[i:i + memory]
            
        return label.reshape(label.size,1), x 
        """
    def calculate_vj_conv(self,alpha,J,memory):
        label, xm = self.data_read(memory)
        sample = xm.shape[0]
        bj = self.calculate_b(alpha, J, memory)
        vj = np.zeros((sample,J))
        for i, xt in enumerate(xm):
            for j, bl in  enumerate(bj):
                vj[i,j] = (xt*bl[::-1]).sum()
        path_vj = os.getcwd() + '/data/vj_train.txt'
        path_yt = os.getcwd() + '/data/yt_train.txt' 
        np.savetxt(path_vj,vj)
        np.savetxt(path_yt,label)
        return label.reshape(label.size,1),vj
