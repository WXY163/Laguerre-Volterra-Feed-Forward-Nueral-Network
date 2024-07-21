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

tranciver = os.getcwd()+'/data/Vdiff_data.txt'
modelpath = os.getcwd()+'/model/modeltrained.pt'
trainerrpath = os.getcwd()+'/log/traerr300.csv'
validateerrpath = os.getcwd()+'/log/valerr300.csv'
figurepath = os.getcwd() + '/trainerror_highspeedlink.png'
outputpath = os.getcwd() + '/log/output_Y300.csv'
labeloutputpath = os.getcwd() + '/log/labeloutput100.csv'


load_tcdata = fd.tranciver_load(tranciver)
memory = 300
input_size = memory
output_size = 1
hidden_size = [memory]
activator  = [mlp.SVN()]
#####0.00001 learning rate for real tranciver samples
learning_rate =0.00001
batch_size = 50
test_sample = 2000
epoch_num = 100
model = mlp.Net(input_size, hidden_size, output_size, activator)
#model.load_state_dict(torch.load(modelpath))

loss_func = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr = learning_rate)


data_Y, data_Xm = load_tcdata.data_read(memory)  
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
#np.savetxt(labeloutputpath, output_Ylabel, delimiter = ',')
#sys.exit('exit')

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
            # train_err.append(loss.item())

            train_err.append(total_loss/(batch_num-1))
            train = 0
        if(i+1)%batch_num == 0:
            validate_err.append(loss.item())
    mlpplot.plotboth(train_err,validate_err,output_Ylabel[1000:3000], output_Y[1000:3000])
    #mlpplot.plotloss(train_err,validate_err)

np.savetxt(trainerrpath, train_err,delimiter=',')
np.savetxt(validateerrpath, validate_err,delimiter=',')
np.savetxt(outputpath, output_Y, delimiter = ',')
torch.save(model.state_dict(),modelpath)

plt.show()


