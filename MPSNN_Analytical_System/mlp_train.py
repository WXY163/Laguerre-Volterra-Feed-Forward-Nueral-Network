import torch.nn as nn
import torch
import torch.nn.functional as tf
import mlp
import numpy as np
import feed_data as fd
import torch.utils.data as utils
import matplotlib.pyplot as plt
import mlp_plot
import os
import sys

dirx = os.getcwd()+'/data/xt_train_save50.csv'
diry = os.getcwd()+'/data/yt_train_save50.csv'
tranciver = os.getcwd()+'/data/data.csv'
modelpath = os.getcwd()+'/model/modeltrained.pt'
trainerrpath = os.getcwd()+'/log/traerr.csv'
validateerrpath = os.getcwd()+'/log/valerr.csv'

Gaussian_data = fd.gaussian_data(True)
#load_data = fd.data_load(dirx,diry)
memory = 50
input_size = memory
output_size = 1
hidden_size = [memory]
activator  = [nn.Tanh()]
learning_rate = 0.001
batch_size = 50
sample = 100000
batch_num = int(99950/50)
test_sample = 2000
epoch_num = 100
model = mlp.Net(input_size, hidden_size, output_size, activator)

loss_func = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr = learning_rate)

data_X,data_Xm = Gaussian_data.xt(sample,memory)
data_Y = Gaussian_data.yt(data_X, memory, 1)
#data_Xm = load_data.loadx(memory)
#data_Y = load_data.loady()

sample = data_Y.size
batch_num = int((sample - memory)/batch_size)

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

np.savetxt(trainerrpath, train_err,delimiter=',')
np.savetxt(validateerrpath, validate_err,delimiter=',')
torch.save(model.state_dict(),modelpath)
plt.show()


