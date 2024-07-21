import numpy as np
import scipy.special as scipy_special
import operator as op
import functools
import matplotlib.pyplot as plt

def ncr(n, r):
    r = min(r, n-r)
    numer = functools.reduce(op.mul, range(n, n-r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r+1), 1)
    return numer//denom

def calculate_bjm(a,j,m):
    elementM = np.arange(m+1)
    elementJ = np.arange(j+1)
    if m < j:
        combineMK = np.zeros(m+1)
        combineJK = np.zeros(m+1)
        for i in range(m+1):
            combineMK[i] = ncr(m,i)
            combineJK[i] = ncr(j,i)
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
            combineMK[i] = ncr(m,i)
            combineJK[i] = ncr(j,i)
        ceff = np.ones(j+1)* -1.0 
        ceff[::2] = 1.0
        powera = np.power(a,elementJ[::-1])
        power1minusa = np.power((1-a), elementJ)
        rn = np.sqrt(a**(m-j))*np.sqrt(1-a)*np.sum(ceff*combineMK*combineJK*powera*power1minusa)
        return rn

def calculate_b(alpha, J, M):
    m = M
    j = J
    b = np.zeros((j,m),np.float32)
    for i in range(j):
        for k in range(m):
            b[i,k] = calculate_bjm(alpha,i,k)
    return b


def calculate_vj(alpha, L, x):
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
    return V
def getM(alpha):
    M = (-30 - np.log(1-alpha))/np.log(alpha)
    return M

a = 0.5
J =5 
M =50 
X = np.ones(50)
X_inv = X[::-1]
b = calculate_b(a, J, M)
np.savetxt("lgparameter.txt",b)
V_conv = np.zeros(b.shape)
V_conv[0,:] = b[0,:]*X
V_conv[1,:] = b[1,:]*X
V_conv[2,:] = b[2,:]*X
V_conv[3,:] = b[3,:]*X
V_conv[4,:] = b[4,:]*X

#V_conv = np.transpose(V_conv)
V = calculate_vj(a, 5, X) 

#print(V_conv)
#print(V)
#plt.plot(b[0])
#plt.plot(b[1])
#plt.plot(b[2])
#plt.plot(b[3])
#plt.plot(b[4])

#plt.show()
