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

memory =40 
J =20 
alpha = 0.855

tranciver = os.getcwd()+'/data/255.txt'
path_Y = os.getcwd() + '/data/yt_train.txt'
path_vj = os.getcwd() + '/data/vj_train.txt'
path_X = os.getcwd() + '/data/xt_train.txt'

modelpath = os.getcwd()+'/model/modeltrained.pt'
trainerrpath = os.getcwd()+'/log/traerr.csv'
validateerrpath = os.getcwd()+'/log/valerr.csv'
"""
for i in range(1,256):
    path = os.getcwd()+'/data/'+str(i)+'.txt'
    load_tcdata = fd.tranciver_load(path)
    label,x_input = load_tcdata.calculate_vj_conv(alpha,J,memory) 
    if i == 1:
        data_Y = label
        data_Xm = x_input
    else:
        data_Y = np.concatenate((data_Y, label))
        data_Xm = np.concatenate((data_Xm, x_input)) 
np.savetxt(path_Y,data_Y)
np.savetxt(path_X,data_Xm)
"""
data_Y_raw = np.loadtxt(path_Y,dtype = np.float32)
data_Y_raw = data_Y_raw.reshape(data_Y_raw.size,1)
data_Xm_raw = np.loadtxt(path_X,dtype = np.float32)
index = np.arange(20400)
np.random.shuffle(index)
data_Y = np.zeros(data_Y_raw.shape)
data_Xm = np.zeros(data_Xm_raw.shape)
for i in range(20400):
    data_Y[i] = data_Y_raw[index[i]]
    data_Xm[i]= data_Xm_raw[index[i]]


input_size = J
output_size = 1
hidden_size = [J]
activator  = [mlp.SVN()]
#activator  = [nn.Tanh()]
batch_size =5 
test_sample = 2000
epoch_num = 500 
learning_rate = 0.00001
model = mlp.Net(input_size, hidden_size, output_size, activator)
model.load_state_dict(torch.load(modelpath))

loss_func = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr = learning_rate)

 
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
            validate_err.append(0.0)
            mlpplot.plotboth(train_err,validate_err,output_Ylabel[10000:10800], output_Y[10000:10800])
    #mlpplot.plotloss(train_err,validate_err)

np.savetxt(trainerrpath, train_err,delimiter=',')
np.savetxt(validateerrpath, validate_err,delimiter=',')
torch.save(model.state_dict(),modelpath)
plt.show()


