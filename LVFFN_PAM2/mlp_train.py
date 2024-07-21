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


tranciver = os.getcwd()+'/data/Vdiff_data.txt'
#path_Yout = os.getcwd() + '/data/yt_system.txt'
#path_Ylabel = os.getcwd() + '/data/yt_label_system.txt'
#path_Yout = os.getcwd() + '/data/yt_channel.txt'
#path_Ylabel = os.getcwd() + '/data/yt_label_channel.txt'
path_Yout = os.getcwd() + '/data/yt_rx.txt'
path_Ylabel = os.getcwd() + '/data/yt_label_rx.txt'
path_lgfilters = os.getcwd()+ '/data/lgfilters8325.txt'
path_vj = os.getcwd() + '/data/vj_train.txt'

#modelpath = os.getcwd()+'/model/modeltrained.pt'
modelpath = os.getcwd() +'/model/j25m300a83channel.pt'
trainerrpath = os.getcwd()+'/log/traerr.csv'
validateerrpath = os.getcwd()+'/log/valerr.csv'

load_tcdata = fd.tranciver_load(tranciver)

memory =300 
J = 25
alpha = 0.83
"""for save laguerre filters"""
load_tcdata.save_b(alpha, J, memory, path_lgfilters)
sys.exit()

input_size = J
output_size = 1
hidden_size = [J]
activator  = [mlp.SVN()]
#activator  = [nn.Tanh()]
batch_size = 50
test_sample = 2000
epoch_num = 150 
learning_rate = 0.001
model = mlp.Net(input_size, hidden_size, output_size, activator)
model.load_state_dict(torch.load(modelpath))
np.savetxt(os.getcwd() + '/data/channelweightlayer1.txt', model.complner[0].weight.data)
np.savetxt(os.getcwd() + '/data/channelbiaslayer1.txt', model.complner[0].bias.data)
np.savetxt(os.getcwd() + '/data/channelweightlayer2.txt', model.complner[2].weight.data)
np.savetxt(os.getcwd() + '/data/channelbiaslayer2.txt', model.complner[2].bias.data)

sys.exit()
loss_func = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr = learning_rate)

data_Y, data_Xm = load_tcdata.calculate_vj_conv(alpha,J,memory)  
#data_Y = np.loadtxt(path_Y)
#data_Y = data_Y.reshape(data_Y.size,1)
#data_Xm = np.loadtxt(path_vj)


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

for epoch in range(epoch_num):
    train  = 1
    total_loss = 0.0

    for i, (x_in, y_label) in enumerate(dataloader):
        output = model(x_in)
        output_Y[i*batch_size:(i+1)*batch_size] = output.data.numpy().flatten()


        loss = loss_func(output,y_label)
        total_loss += loss.item() 
        if(train): 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(i+1)%(batch_num-1) == 0:
            train_err.append(total_loss/(batch_num-1))
            train = 0
        if(i+1)%batch_num == 0:
            validate_err.append(loss.item())
            mlpplot.plotboth(train_err,validate_err,output_Ylabel[1000:3000], output_Y[1000:3000])
    #mlpplot.plotloss(train_err,validate_err)

np.savetxt(path_Yout,output_Y)
np.savetxt(path_Ylabel,output_Ylabel)

np.savetxt(validateerrpath, validate_err,delimiter=',')
torch.save(model.state_dict(),modelpath)
plt.show()


