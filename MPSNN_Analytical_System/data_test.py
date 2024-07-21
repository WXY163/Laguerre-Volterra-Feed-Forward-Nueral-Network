import numpy as np
import os 
import sys

from matplotlib import pyplot as plt

import feed_data as fd

dirx = os.getcwd()+'/data/xt_train_save50.csv'
diry = os.getcwd()+'/data/yt_train_save50.csv'
dir_testx = os.getcwd()+'/data/xt_test.csv'
dir_testy = os.getcwd()+'/data/yt_test_save50.csv'
data_dir = os.getcwd()+'/data/data.csv'

load_data = fd.tranciver_load(data_dir)


memory =100 
x, y = load_data.data_read(memory)
#plt.hist(data_X)
#plt.figure()
#plt.hist(data_testX)
#plt.show()
plt.plot(x)
plt.show()








