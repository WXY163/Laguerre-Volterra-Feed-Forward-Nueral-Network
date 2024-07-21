import torch
import torch.nn as nn
import torch.nn.functional as F
class SVN(nn.Module):
    def forward(ctx, input):
        power =3 
        input_rtn = input.clone()
        return input_rtn.pow(power)
"""
    def backward(ctx, grad_output):
        power =2
        grad_input =torch.FloatTensor(grad_output.shape).zero_() 
        saved_input, = ctx.saved_tensors
        grad = saved_input.pow(power - 1)
        grad = torch.mul(grad, grad_output)
        return torch.mul(grad,power) 
"""





class Net(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, activator):
        super(Net,self).__init__()
        layer = []
        nextlen = input_size
        for len_layerk, actk in zip(hidden_size,activator):#hidsz and activ must be of same length
            fc = nn.Linear(nextlen,len_layerk)
            layer += [fc,actk]#append a fully connected layer followed by an activation
            
            nextlen = len_layerk#update inlen of next layer
        
        #last layer to output
        layer += [nn.Linear(nextlen,output_size) ]
        self.complner = nn.Sequential(*layer)

        
    def forward(self, InVec):
        # print('Input to forward: ', InVec.dtype, ' is cuda: ', InVec.is_cuda)
        output = self.complner(InVec)
        return output

