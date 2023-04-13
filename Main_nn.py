#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:41:12 2023

@author: evanesauter

Main for Neural Networks
"""
#Imports
import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#File Imports
import HelperFunctions as hf
from HelperFunctions import Results
import MineLSTM as mylstm
import MineGRU as mygru

"""Explaining the layout of the Excel""" 
train_idx = 2 #Train City Index
test_idx = 2 #Test City Index
City_dict = {0:'Amherst',1:'Davis',2:'Huron',3:'Santa Barbara',4:'La Jolla',
           5:'Davis 6yr',6:'Huron 6yr',7:'Santa Barbara 6yr',8:'La Jolla 6yr'}

#Type of normalization
Norm_dict = {0:'Min-Max',1:'Z-score',2:'Decimal'}
norm_idx = 0


"""Loading of City Data"""
#Importing/Loading Data
df_City3Year = pd.read_excel("Further Consolidated Data, HnL.xlsx", sheet_name = train_idx)
df_City3Year = df_City3Year.dropna()
df_City3Year2 = pd.read_excel("Further Consolidated Data, HnL.xlsx", sheet_name = test_idx)
df_City3Year2 = df_City3Year2.dropna()


"""Generating the Heatmap"""
df_corr = df_City3Year.corr(method='kendall')
mask = np.triu(np.ones_like(df_corr, dtype=bool))
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(df_corr, mask=mask, cmap='mako_r', vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={"shrink": .8});


"""Miscellaneous Setup"""
#The size of the testing batch. Training size is 1 - {this value}
TestBatch_Size = 0.3

#Values necessary for the Sliding Window
Window_Size = 11
Data_dict_train = {} #Dict that the windows of X data, with output Y, will be stored in
Data_dict_test = {} #Dict that the windows of X data, with output Y, will be stored in
path_train = './Data_train/'
path_test = './Data_test/'


#Array for the Random State
Random_Seed = [0,10,20] #Three known ints for reproducability

Column_names = ['Year','Month','Day','Hour','Minute','DHI','DNI','GHI','Clearsky DHI','Clearsky DNI',
                'Clearsky GHI','Cloud Type','Dew Point','Solar Zenith Angle','Surface Albedo','Wind Speed',
                'Precipitable Water','Wind Direction','Relative Humidity','Temperature','Pressure','Output Power']


"""Preprocessing of Data"""
#Normalization method
if Norm_dict[norm_idx] == 'Min-Max':
    df_City3Year, Scalers = hf.Scale_Columns(df_City3Year,Columns=Column_names,Scalers=None)
    df_City3Year2, Scalers = hf.Scale_Columns(df_City3Year2,Columns=Column_names,Scalers=Scalers)
elif Norm_dict[norm_idx] == 'Z-score':
    df_City3Year = df_City3Year.apply(stats.zscore)
    df_City3Year2 = df_City3Year2.apply(stats.zscore)
elif Norm_dict[norm_idx] == 'Decimal':
    df_City3Year = hf.Decimal_Scale(df_City3Year, Column_names)
    df_City3Year2 = hf.Decimal_Scale(df_City3Year2, Column_names)


#Sequentially splitting the data into train & test
X_train, X_test, Y_train, Y_test = hf.Sequential_Split(df_City3Year, TestBatch_Size) #Splits for first city
X_train2, X_test2, Y_train2, Y_test2 = hf.Sequential_Split(df_City3Year2, TestBatch_Size) #Splits for second city

#The sliding window & pickling
hf.save_pkl(X_train,Y_train,Data_dict_train,Window_Size,path_train) #Slides through & pickles the training set
hf.save_pkl(X_test2,Y_test2,Data_dict_test,Window_Size,path_test) #Slides through & pickles the testing set


#Clear the text files of results
hf.remove_all_results()

#Running the main of a given neural network
for k, i in enumerate(Random_Seed):
    mylstm.main(i)
    mygru.main(i)
    print('-- -- Iteration {} Complete -- --\n'.format(k+1))


#Notifies which city & normaliztion method were used
print('Trained model on {} data, Tested on {}\n'.format(City_dict[train_idx],City_dict[test_idx]))
print('Normalization Method: {}'.format(Norm_dict[norm_idx]))

"""Printing MSE results, the mu, & STDev"""
print('LSTM MSE mu & std:', hf.mustd(hf.read_results(Results.path_result + Results.lstm_ts)))
print('GRU MSE mu & std:', hf.mustd(hf.read_results(Results.path_result + Results.gru_ts)))

"""Printing the r2 scores and mu & STDev"""
print('LSTM r2, mu, & std:', hf.mustd(hf.read_results(Results.path_result + Results.lstm_r2)))
print('GRU r2, mu, & std:', hf.mustd(hf.read_results(Results.path_result + Results.gru_r2)))



