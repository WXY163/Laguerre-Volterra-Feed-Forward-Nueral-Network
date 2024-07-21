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


tranciver = os.getcwd()+'/data/pam4_old.txt'
path_Y = os.getcwd() + '/data/yt_train.txt'
path_vj = os.getcwd() + '/data/vj_train.txt'

modelpath = os.getcwd()+'/model/modeltrained.pt'
trainerrpath = os.getcwd()+'/log/traerr.csv'
validateerrpath = os.getcwd()+'/log/valerr.csv'

load_tcdata = fd.tranciver_load(tranciver)

memory =150 
J =10 
alpha = 0.91
input_size = J
output_size = 1
hidden_size = [J]
activator  = [mlp.SVN()]
#activator  = [nn.Tanh()]
batch_size = 50
test_sample = 2000
epoch_num = 100 
learning_rate = 0.0007
model = mlp.Net(input_size, hidden_size, output_size, activator)
#model.load_state_dict(torch.load(modelpath))

loss_func = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr = learning_rate)

data_Y, data_Xm = load_tcdata.calculate_vj(alpha,J)  
#data_Y = np.loadtxt(path_Y)
#data_Y = data_Y.reshape(data_Y.size,1)
#data_Xm = np.loadtxt(path_vj)
#sys.exit()

data_Y = data_Y[100:]
data_Xm = data_Xm[100:]
sample = data_Y.size
batch_num = int((sample)/batch_size)
validate_num = int(sample*0.3)
validate_batch_num = int(validate_num/batch_size)


tensor_x = torch.stack([torch.Tensor(i) for i in data_Xm])
tensor_y = torch.stack([torch.Tensor(i) for i in data_Y])

tensor_dataset = utils.TensorDataset(tensor_x,tensor_y)
dataloader = utils.DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=False)

mlpplot = mlp_plot.mlp_plot()
train_err = []
validate_err = []
output_Y = np.zeros(sample)
output_Ylabel = data_Y[:sample] 

for epoch in range(epoch_num):
    train_loss = 0.0
    validate_loss = 0.0
    train_count = 0
    for i, (x_in, y_label) in enumerate(dataloader):
        
        train = 1
        output = model(x_in)
        output_Y[i*batch_size:(i+1)*batch_size] = output.data.numpy().flatten()

        loss = loss_func(output,y_label)
        if (i) <= validate_batch_num -1:
            train = 0 
        if train == 1:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  
            train_count +=   1
        else:
            validate_loss += loss.item()
        if i!=0 and i == (validate_batch_num -1):
            validate_err.append(validate_loss/validate_batch_num)
        if i!=0 and i%(batch_num -1) ==0:
            train_err.append(train_loss/train_count)

    mlpplot.plotboth(train_err,validate_err,output_Ylabel[1000:3000], output_Y[1000:3000])
    #mlpplot.plotloss(train_err,validate_err)
#np.savetxt("inferout.txt",output_Y)
#np.savetxt("reference.txt",output_Ylabel)
np.savetxt(trainerrpath, train_err,delimiter=',')
np.savetxt(validateerrpath, validate_err,delimiter=',')
torch.save(model.state_dict(),modelpath)
plt.show()


