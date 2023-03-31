#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:12:04 2023

Sourced a classification version of this model from:
    https://github.com/pytorch/examples/blob/main/mnist_rnn/main.py
The main contibutors to that document were:
    Rakesh Malviya & Steven Liu    

@author: evanesauter

Drafting an LSTM
"""

#Imports
from __future__ import print_function
from sklearn.metrics import r2_score
import HelperFunctions as hf
from HelperFunctions import Results

#Imports for LSTM
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=64, num_layers=3, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.50)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.fc3 = nn.Linear(10, 1)         
       


    def forward(self, input):
        input = input.reshape(-1, 11 ,20)
        batch_size = input.size(0)
        h0 = torch.randn(3,batch_size,64)
        c0 = torch.randn(3,batch_size,64)
        
        input = input.clone().to(torch.float32) #converts from float64 to float32
        output, (_,_) = self.lstm(input,(h0,c0))    

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        output = output[:, -1, :]
        output = self.batchnorm(output)
        output = F.relu(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = F.relu(output)         
        output = self.dropout2(output)  
        output = self.fc2(output)
        output = F.relu(output)         
        output = self.dropout3(output)  
        output = self.fc3(output)
        return output


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        feature = data['feature']
        target = data['target']
        target_adjusted_shape = (len(target),1)
        target = target.view(target_adjusted_shape)
        optimizer.zero_grad()
        model = model.float()
        output = model(feature)
        loss = hf.loss_fn(output, target)
        loss = loss.to(torch.float32)
        loss.backward()                         
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(feature), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, test_loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            feature = test_data['feature']
            target = test_data['target']
            target_adjusted_shape = (len(target),1)
            target = target.view(target_adjusted_shape)
            model = model.float()
            output = model(feature)
    
            test_loss += hf.loss_fn(output, target).item() * target.size(0)
            
            if args.dry_run:
                break
    
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    if epoch == (args.epochs - 1):
        R2_LSTM = r2_score(target,output)
        hf.save_result(R2_LSTM, Results.path_result + Results.lstm_r2)
        hf.save_result(test_loss, Results.path_result + Results.lstm_ts)
        
    

def main(rand_seed):
    # Training settings, adjust to tune the hyperparameters
    parser = argparse.ArgumentParser(description='LSTM using RNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',          #Train Batch Size
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',   #Test Batch Size
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',              #Num Epochs
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.99, metavar='LR',             #Learning Rate
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',            #Gamma
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,            #Use Cuda?
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,            #Dry Run?
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=rand_seed, metavar='S',         #Random Seed
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',        #Num batches before train
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,         #Save the current model?
                        help='for Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(hf.CustomDataset(is_train=True), batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(hf.CustomDataset(is_train=False), batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(args, model, test_loader, epoch)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./Model_Output/lstm.pt")
