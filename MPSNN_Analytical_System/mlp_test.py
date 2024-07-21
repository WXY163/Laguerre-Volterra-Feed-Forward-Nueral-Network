import numpy as np
import torch.nn as nn

import torch
import mlp
import os
import matplotlib.pyplot as plt
import feed_data as fd
import torch.utils.data as utils
tranciver = os.getcwd()+'/data/Vdiff_data.txt'
load_tcdata = fd.tranciver_load(tranciver)

memory = 300
test_sample = 2000
input_size = memory
output_size = 1
hidden_size = [memory]
activator  = [nn.Tanh()]
model = mlp.Net(input_size, hidden_size, output_size, activator)
modelpath = os.getcwd()+'/model/modeltrained.pt'
loss_func = nn.MSELoss()
#Gaussian_testdata = fd.gaussian_data(True,data_type='test') 
model.load_state_dict(torch.load(modelpath))

#data_testX,data_testXm = Gaussian_testdata.xt(test_sample,memory)
#data_testY = Gaussian_testdata.yt(data_testX, memory)
data_Y, data_Xm = load_tcdata.data_read(memory)  
data_testY = data_Y[-1*test_sample:]
data_testXm = data_Xm[-1*test_sample:]
tensor_testx = torch.stack([torch.Tensor(i) for i in data_testXm])
tensor_testy = torch.stack([torch.Tensor(i) for i in data_testY])

tensor_testdataset = utils.TensorDataset(tensor_testx,tensor_testy)
dataloader_test = utils.DataLoader(dataset=tensor_testdataset)
test_err = []
output_Y = np.zeros(test_sample)
for i, (x_in, y_label) in enumerate(dataloader_test):
    output = model(x_in)
    output_Y[i] = output.data
    loss = loss_func(output,y_label)
    test_err.append(loss.item())
plt.plot(test_err)
plt.ylim(0,0.01)
plt.figure()
plt.plot(data_testY,'b-')
plt.plot(output_Y,'r--')

plt.show()
