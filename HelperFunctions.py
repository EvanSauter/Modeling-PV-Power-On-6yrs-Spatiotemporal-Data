#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:36:52 2023

@author: evanesauter

Helper Functions
"""

import torch
import pickle
import glob
import os
import shutil
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from statistics import mean
from statistics import stdev
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



#Paths for results of models
class Results():
    path_result = './Results/'
    #The files that hold the respective results
    hmm_ts = 'hmm_ts.txt'   #test scores
    hmm_r2 = 'hmm_r2.txt'   #r2 scores
    lstm_ts = 'lstm_ts.txt' #These will be added to the path
    lstm_r2 = 'lstm_r2.txt'
    gru_ts = 'gru_ts.txt'
    gru_r2 = 'gru_r2.txt'
    mlp_ts = 'mlp_ts.txt'
    mlp_r2 = 'mlp_r2.txt'


"""Sequential Data Spliting"""
def Sequential_Split(df_City,test_size):
    '''
    Takes in the dataset and the decimal value dictating how large the testing size will be
    Returns the sequentially split data
    '''
    #Splitting Data into Test & Train
    df_City_Train = df_City.iloc[0:round(len(df_City)*(1-test_size)),:]
    df_City_Test = df_City.iloc[round(len(df_City)*(1-test_size)):round(len(df_City)),:]
    
    #Splitting the data into X & Y of respective training and testing
    Xtrain = df_City_Train.drop(labels=['Output Power'],axis=1) #Creates the X train values
    Ytrain = df_City_Train['Output Power'].values #creates Y train values
    Xtest = df_City_Test.drop(labels=['Output Power'],axis=1) #Creates the X test values
    Ytest = df_City_Test['Output Power'].values #creates Y test values
    return Xtrain, Xtest, Ytrain, Ytest


"""Splitting Data by Random Sampling"""
def Random_Split(df_City,Test_Size,Rand_State):
    '''
    Takes in the dataset and the decimal value dictating how large the testing size will be
    Returns the randomly split data 
    '''
    #Splitting the data into X & Y of respective training and testing
    X = df_City.drop(labels=['Output Power'],axis=1) #Creates the X test values
    Y = df_City['Output Power'].values #creates Y test values
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=Test_Size,random_state=Rand_State)    
    return Xtrain, Xtest, Ytrain, Ytest


"""Mean & Standard Deviation"""
def mustd (Mdl_Scores):
    '''
    Takes in an array
    Returns the array with the mean and standard deviation appended in that order
    '''
    Mdl_Scores_mu = mean(Mdl_Scores)
    Mdl_Scores_std = stdev(Mdl_Scores, Mdl_Scores_mu)
    Mdl_Scores.append(Mdl_Scores_mu)
    Mdl_Scores.append(Mdl_Scores_std)
    return Mdl_Scores


"""Normalizing & Scaling the Data"""
def Scale_Columns(df_City,Columns,Scalers):
    '''
    Takes in the dataframe of data
    returns the normalized dataframe based, the testing set scaled by the training set
    '''
    if Scalers is None:
        Scalers = {}
        for col in Columns:
            Scaler = MinMaxScaler().fit(df_City[[col]])
            df_City[col] = Scaler.transform(df_City[[col]])
            Scalers[col] = Scaler
    else:
        for col in Columns:
            Scaler = Scalers.get(col)
            df_City[col] = Scaler.transform(df_City[[col]])
    return df_City,Scalers   


"""Sliding Window Function"""
def save_pkl(df_X,df_Y,Dict,period,path):
    '''
    Layout of inputtted dataframe:
    df_X = | time related features | weather realated features |
    df_Y = | output power |
    '''
    
    '''
    Creating the sliding window
    '''
    for i in range (0,len(df_X) - period):
        # col = # of features = # number of timestep you want as the lstm input (period)
        feature = df_X.iloc[i:i+period,1:21] #a row by column matrix
        regression_target = df_Y[i+period-1] # obtains the Output Power at index
        Dict = {'feature': feature, 'target': regression_target}
        
        #Save dict as a pickle
        save_dict_pkl(Dict, f'{path}{i}.pkl')
    return None


"""Saving the Dictionary via Pickling"""
def save_dict_pkl(data,path):
    '''
    Saves the dictionary as a pkl
    '''
    with open(path,'wb') as f:
        pickle.dump(data,f) #Saves the pickle entry to the file
    
    return None


def read_pkl(path):
    '''
    Takes in the path to the saved dict of pkl
    returns the pickle
    '''
    with open(path, 'rb') as pkl:
        
        return pickle.load(pkl) #reads the dict from pickle


def df2t(feature):
    '''
    Takes in the key of the dict
    returns the key, the dataframe, as a tensor
    '''
    temp = feature.to_numpy()
    tnsr = torch.from_numpy(temp)
    return tnsr


def f2ft(target):
    '''
    Take in the value of the dict (a numpy float)
    returns the value, the Y output power, as float64 tensor
    '''
    ft = torch.tensor(target, dtype=torch.double)
    return ft 


"""Custom Dataset Creation, for NN"""
class CustomDataset(Dataset):
    '''
    Creates a dataset based on the data contained within the pkl files
    Returns the pkl at a given instance as a tensor
    '''
    def __init__(self, is_train=None):
        
        if is_train:
            self.files = glob.glob('./Data_train/*.pkl')
            
        else:
            self.files = glob.glob('./Data_test/*.pkl')
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        Dict = read_pkl(self.files[idx])
        feature = df2t(Dict['feature']) #converts dataframe -> tensor 
        target = f2ft(Dict['target']) #converts float - > float tensor
        return {'feature': feature.float(), 'target': target.float()}


def loss_fn(pre, tar):
    '''
    The MSE Loss function
    '''
    loss = nn.MSELoss()
    return loss(pre, tar)


def save_result(data, path):
    '''
    Appends the result to the given file
    '''
    with open(path,'a') as f:
        print(data, file=f) #Saves the pickle entry to the file
    return None


def read_results(path):
    '''
    Opens and reads the results, returns as list?
    '''
    with open(path, 'r') as f:
        results = [float(i) for i in f.read().split()]
        #results = [ int(i) for i in f]
        return results


def remove_all_results():
    '''
    Deletes the Results directory, replaces with empty directory
    Should be called before the main()
    '''
    path = Results.path_result
    if os.path.isdir(path):
        shutil.rmtree(Results.path_result) #Deletes entire directory (folder)
        os.mkdir(Results.path_result) #Creates entire Directory
    else:
        os.mkdir(Results.path_result) #Creates entire Directory

    return None


def random_seed(seed_value):
    '''
    When necessary, supplies the seed from the value or array inserted
    '''
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
