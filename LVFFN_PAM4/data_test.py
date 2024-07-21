import numpy as np
import os 
import sys

from matplotlib import pyplot as plt

import feed_data as fd
"""
dirx = os.getcwd()+'/data/xt_train_save50.csv'
diry = os.getcwd()+'/data/yt_train_save50.csv'
dir_testx = os.getcwd()+'/data/xt_test.csv'
dir_testy = os.getcwd()+'/data/yt_test_save50.csv'
data_dir = os.getcwd()+'/data/data.csv'

load_data = fd.tranciver_load(data_dir)
"""
gd = fd.gaussian_data()
memory =10 
alpha = 0.2
J = 4
samples =100
bj = gd.calculate_b(alpha,J,memory)
#m = np.sum(bj[1]*bj[0])
#print(m)
vj = gd.calculate_vj_conv(alpha, J, memory, samples)
#vj = gd.calculate_vj_recur(alpha, J, memory, samples)

#plt.plot(bj[0])
#plt.plot(bj[1])
#plt.plot(bj[2])
#plt.plot(bj[3])
#plt.plot(vj)
#print(vj.shape)
plt.show()








