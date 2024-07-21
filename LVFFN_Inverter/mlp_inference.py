import torch.nn as nn
import torch
import torch.nn.functional as tf
import mlp
import numpy as np
import feed_data as fd
import torch.utils.data as utils
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import mlp_plot
import os
import sys


tranciver = os.getcwd()+'/data/pam4.txt'
path_Y = os.getcwd() + '/data/yt_train.txt'
path_vj = os.getcwd() + '/data/vj_train.txt'

modelpath = os.getcwd()+'/model/modeltrained.pt'
trainerrpath = os.getcwd()+'/log/traerr.csv'
validateerrpath = os.getcwd()+'/log/valerr.csv'

load_tcdata = fd.tranciver_load(tranciver)

memory =300 
J =20 
alpha = 0.9
input_size = J
output_size = 1
hidden_size = [J]
activator  = [mlp.SVN()]
#activator  = [nn.Tanh()]
batch_size = 50
test_sample = 2000
epoch_num = 100 
learning_rate = 0.0005
model = mlp.Net(input_size, hidden_size, output_size, activator)
model.load_state_dict(torch.load(modelpath))

loss_func = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr = learning_rate)

data_Y, data_Xm = load_tcdata.calculate_vj_conv(alpha,J,memory)  
#data_Y = np.loadtxt(path_Y)
#data_Y = data_Y.reshape(data_Y.size,1)
#data_Xm = np.loadtxt(path_vj)
#sys.exit()


sample = data_Y.size
batch_num = int((sample)/batch_size)

tensor_x = torch.stack([torch.Tensor(i) for i in data_Xm[:sample]])
tensor_y = torch.stack([torch.Tensor(i) for i in data_Y[:sample]])

tensor_dataset = utils.TensorDataset(tensor_x,tensor_y)
dataloader = utils.DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=False)

mlpplot = mlp_plot.mlp_plot()
train_err = []
validate_err = []
output_Y = np.zeros(sample)
output_Ylabel = data_Y[:sample] 
total_loss = 0
for i, (x_in, y_label) in enumerate(dataloader):
    output = model(x_in)
    output_Y[i*batch_size:(i+1)*batch_size] = output.data.numpy().flatten()


    loss = loss_func(output,y_label)
    total_loss += loss.item() 
mlpplot.plotboth(train_err,validate_err,output_Ylabel[0:3000], output_Y[0:3000])
    #mlpplot.plotloss(train_err,validate_err)

np.savetxt(trainerrpath, train_err,delimiter=',')
np.savetxt(validateerrpath, validate_err,delimiter=',')
torch.save(model.state_dict(),modelpath)
plt.show()


