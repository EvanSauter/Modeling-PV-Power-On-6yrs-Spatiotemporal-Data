#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:25:08 2023

Sourced a classification version of this model from:
    https://github.com/emadRad/lstm-gru-pytorch
The main contributor of that document was:
    Emad Bahrami

@author: evanesauter

My version of GRU
"""

#Initial Imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

#Additional from other files
import HelperFunctions as hf
from HelperFunctions import Results
from torchmetrics import R2Score
from sklearn.metrics import r2_score


 
'''
STEP 1: LOADING DATASET
'''

batch_size = 64
test_batch_size = 1000
num_epochs = 14
train_loader = torch.utils.data.DataLoader(hf.CustomDataset(is_train=True), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(hf.CustomDataset(is_train=False), batch_size=test_batch_size, shuffle=True, drop_last=True)

Rand_Seed = [0,10,20]

class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()  

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
    
        return hy
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
         
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)) 
       
        outs = []
        
        hn = h0[0,:,:]
        
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs.append(hn)

        out = outs[-1].squeeze()
        
        out = self.fc(out)

        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 20
hidden_dim = 256
layer_dim = 2
output_dim = 1

model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

 
'''
STEP 5: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate) 
'''
STEP 6: TRAIN & TEST THE MODEL
'''
 
# Number of steps to unroll
seq_dim = 11 #Size of window

def main(rand_seed):
    iter = 0
    test_loss = 0
    model.eval()
    
    hf.random_seed(rand_seed) 
    
    for epoch in range(num_epochs):
        for i,data in enumerate(train_loader):
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            feature = data['feature']
            target = data['target']
            
            feature = Variable(feature.reshape(-1, seq_dim, input_dim))
            target = Variable(target)
            target_adjusted_shape = (len(target),1)
            target = target.view(target_adjusted_shape)
        
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
             
            # Forward pass to get output/logits
            output = model(feature)
    
            # Calculate Loss: softmax --> cross entropy loss
            loss = hf.loss_fn(output, target)
    
            # Getting gradients w.r.t. parameters
            loss.backward()
    
            # Updating parameters
            optimizer.step()
            
            iter += 1
            
            #Test 
            if iter % 250 == 0:
                test_loss = 0
                # Iterate through test dataset
                with torch.no_grad():
                    for i,test_data in enumerate(test_loader):
                        feature = test_data['feature']
                        target = test_data['target']
                        
                        feature = Variable(feature.reshape(-1, seq_dim, input_dim))
                        target_adjusted_shape = (len(target),1)
                        target = target.view(target_adjusted_shape)
                        target = Variable(target)
                        
                        # Forward pass only to get logits/output
                        output = model(feature)
                        
                        test_loss += hf.loss_fn(output, target).item() * target.size(0)
                    
                test_loss /= len(test_loader.dataset)
                
                print('Iteration: {} Test Loss: {}'.format(iter, test_loss))
                
    if epoch == (num_epochs - 1): #num_epochs - 1 because range() is used
        #Removing the grad from the tensors to be able to call numpy, prep for r2 score
        target = target.detach().numpy()
        output = output.detach().numpy()
        #Obtaining the Scores
        R2_GRU = r2_score(target,output)
        hf.save_result(R2_GRU.item(), Results.path_result + Results.gru_r2)
        hf.save_result(test_loss, Results.path_result + Results.gru_ts)

