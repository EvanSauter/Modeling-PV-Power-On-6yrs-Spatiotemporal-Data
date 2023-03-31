#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:29:59 2022

@author: evanesauter

Simple Testing of Differen Algorithms
"""

#Imports
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import HelperFunctions as hf

#Imports for Models, in order of appearance
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


"""Explaining the layout of the Excel""" 
train_idx = 2 #Train City Index
test_idx = 2 #Test City Index
City_dict = {0:'Amherst',1:'Davis',2:'Huron',3:'Santa Barbara',4:'La Jolla',
           5:'Davis 6yr',6:'Huron 6yr',7:'Santa Barbara 6yr',8:'La Jolla 6yr'}

"""Loading of City Data"""
#Importing/Loading Data
df_City3Year = pd.read_excel("Further Consolidated Data, HnL.xlsx", sheet_name = train_idx)
df_City3Year = df_City3Year.dropna()
df_City3Year2 = pd.read_excel("Further Consolidated Data, HnL.xlsx", sheet_name = test_idx)
df_City3Year2 = df_City3Year2.dropna()

#Change the below if looking to perform sequential data splitting, or randomized splitting
Sequential = False

#Change the below to adjust the size of size of testing data
TestBatch_Size = 0.3 

Column_names = ['Year','Month','Day','Hour','Minute','DHI','DNI','GHI','Clearsky DHI','Clearsky DNI',
                'Clearsky GHI','Cloud Type','Dew Point','Solar Zenith Angle','Surface Albedo','Wind Speed',
                'Precipitable Water','Wind Direction','Relative Humidity','Temperature','Pressure','Output Power']

#Normalizing the entire Dataframe
df_City3Year, Scalers = hf.Scale_Columns(df_City3Year,Columns=Column_names,Scalers=None)
df_City3Year2, Scalers = hf.Scale_Columns(df_City3Year2,Columns=Column_names,Scalers=Scalers)


#Array for the Random State
Random_Seed = [0,10,20] 

#Arrays for each model to hold the R2 scores, and later the mean and std
Scores_LinRegr_r2 = []
Scores_Ridge_r2 = []
Scores_SGDregr_r2 = []
Scores_PLSr_r2 = []
Scores_KernelRidge_r2 = []
Scores_RF_r2 = []
Scores_DT_r2 = []
Scores_SVR_r2 =[]
Scores_NuSVR_r2 = []
Scores_MLP_NN_r2 = []
Scores_KNeighbor_r2 = []
Scores_Bagging_r2 = []
Scores_ElasticNet_r2 = []

#Lists for the MSE scores
Scores_LinRegr_mse = []
Scores_Ridge_mse = []
Scores_SGDregr_mse = []
Scores_PLSr_mse = []
Scores_KernelRidge_mse = []
Scores_RF_mse = []
Scores_DT_mse = []
Scores_SVR_mse =[]
Scores_NuSVR_mse = []
Scores_MLP_NN_mse = []
Scores_KNeighbor_mse = []
Scores_Bagging_mse = []
Scores_ElasticNet_mse = []

#Lists for the MSE scores
Scores_LinRegr_rmse = []
Scores_Ridge_rmse = []
Scores_SGDregr_rmse = []
Scores_PLSr_rmse = []
Scores_KernelRidge_rmse = []
Scores_RF_rmse = []
Scores_DT_rmse = []
Scores_SVR_rmse =[]
Scores_NuSVR_rmse = []
Scores_MLP_NN_rmse = []
Scores_KNeighbor_rmse = []
Scores_Bagging_rmse = []
Scores_ElasticNet_rmse = []


"""Iterates through all the models"""
for k, i in enumerate(Random_Seed):
    
    
    """Preprocessing of Data"""
    if Sequential is True: 
        #Splitting Data Sequentially, uncomment when necessary
        X_train, X_test, Y_train, Y_test = hf.Sequential_Split(df_City3Year, TestBatch_Size) #Splits for first city
        X_train2, X_test2, Y_train2, Y_test2 = hf.Sequential_Split(df_City3Year2, TestBatch_Size) #Splits for second city
    else:
        #Randomly Splits the data
        X_train, X_test, Y_train, Y_test = hf.Random_Split(df_City3Year, TestBatch_Size, i) #Splits for first city
        X_train2, X_test2, Y_train2, Y_test2 = hf.Random_Split(df_City3Year2, TestBatch_Size, i) #Splits for second city
   
    
    """Regression Models"""
    """Linear Regresion Models & Kernel Ridge"""
    #Linear Regression
    Mdl_LinRegr = LinearRegression(fit_intercept=True,positive=False).fit(X_train, Y_train)
    
    R2_LinRegr = r2_score(Y_test2, Mdl_LinRegr.predict(X_test2)) #r2 score
    mse_LinRegr = mean_squared_error(Y_test2, Mdl_LinRegr.predict(X_test2)) #MSE
    rmse_LinRegr = mean_squared_error(Y_test2, Mdl_LinRegr.predict(X_test2), squared=False) #RMSE
    
    Scores_LinRegr_r2.append(R2_LinRegr)
    Scores_LinRegr_mse.append(mse_LinRegr)
    Scores_LinRegr_rmse.append(rmse_LinRegr)
    
    #Ridge
    Mdl_Ridge = Ridge(alpha=0.1,fit_intercept=True,max_iter=None,tol=1e-3,solver='svd',positive=False,random_state=0).fit(X_train,Y_train)
    
    R2_Ridge = r2_score(Y_test2, Mdl_Ridge.predict(X_test2)) #r2 score
    mse_Ridge = mean_squared_error(Y_test2, Mdl_Ridge.predict(X_test2)) #MSE
    rmse_Ridge = mean_squared_error(Y_test2, Mdl_Ridge.predict(X_test2), squared=False) #RMSE
   
    Scores_Ridge_r2.append(R2_Ridge)
    Scores_Ridge_mse.append(mse_Ridge)
    Scores_Ridge_rmse.append(rmse_Ridge)
    
    #Stochastic Gradient Descent
    Mdl_SGDregr = SGDRegressor(loss='squared_error',penalty='elasticnet',alpha=1e-4,l1_ratio=0.5,fit_intercept=True,max_iter=1000,tol=1e-3,shuffle=True,epsilon=0.1,random_state=0,learning_rate='invscaling',eta0=0.07,power_t=0.25,average=11).fit(X_train, Y_train) 
    
    R2_SGDregr = r2_score(Y_test2, Mdl_SGDregr.predict(X_test2)) #r2 score
    mse_SGDregr = mean_squared_error(Y_test2, Mdl_SGDregr.predict(X_test2)) #MSE
    rmse_SGDregr = mean_squared_error(Y_test2, Mdl_SGDregr.predict(X_test2), squared=False) #RMSE
    
    Scores_SGDregr_r2.append(R2_SGDregr)
    Scores_SGDregr_mse.append(mse_SGDregr)
    Scores_SGDregr_rmse.append(rmse_SGDregr)
    
    #Partial Least Squares Regression
    Mdl_PLSr = PLSRegression(n_components=15,scale=True,max_iter=1000,tol=1e-6,copy=True).fit(X_train,Y_train)
    
    R2_PLSr = r2_score(Y_test2, Mdl_PLSr.predict(X_test2)) #r2 score
    mse_PLSr = mean_squared_error(Y_test2, Mdl_PLSr.predict(X_test2)) #MSE
    rmse_PLSr = mean_squared_error(Y_test2, Mdl_PLSr.predict(X_test2), squared=False) #RMSE
    
    Scores_PLSr_r2.append(R2_PLSr)
    Scores_PLSr_mse.append(mse_PLSr)
    Scores_PLSr_rmse.append(rmse_PLSr)
    
    #Kernel Ridge
    Mdl_KernelRidge = KernelRidge(alpha=1.0,kernel='poly',degree=10).fit(X_test,Y_test) #having gamma=1.5 boosts the score but throws a warning

    R2_KR = r2_score(Y_test2, Mdl_KernelRidge.predict(X_test2)) #r2 Score
    mse_KR = mean_squared_error(Y_test2, Mdl_KernelRidge.predict(X_test2)) #MSE
    rmse_KR = mean_squared_error(Y_test2, Mdl_KernelRidge.predict(X_test2), squared=False) #RMSE
    
    Scores_KernelRidge_r2.append(R2_KR)
    Scores_KernelRidge_mse.append(mse_KR)
    Scores_KernelRidge_rmse.append(rmse_KR)
    
    """Random Forest & Decision Tree Regressors"""
    #Random Forest
    Mdl_RF = RandomForestRegressor(n_estimators=100,criterion='squared_error',min_samples_leaf=17,max_features=1.0,random_state=0).fit(X_train, Y_train)
    
    R2_RF = r2_score(Y_test2, Mdl_RF.predict(X_test2)) #r2 score
    mse_RF = mean_squared_error(Y_test2, Mdl_RF.predict(X_test2)) #MSE
    rmse_RF = mean_squared_error(Y_test2, Mdl_RF.predict(X_test2), squared=False) #RSME
    
    Scores_RF_r2.append(R2_RF)
    Scores_RF_mse.append(mse_RF)
    Scores_RF_rmse.append(rmse_RF)
   
    #Decision Tree
    Mdl_DT = DecisionTreeRegressor(criterion='squared_error',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=70,max_features=1.0,random_state=0).fit(X_train, Y_train)
    
    R2_DT = r2_score(Y_test2, Mdl_DT.predict(X_test2)) #r2 score
    mse_DT = mean_squared_error(Y_test2, Mdl_DT.predict(X_test2)) #MSE
    rmse_DT = mean_squared_error(Y_test2, Mdl_DT.predict(X_test2), squared=False) #RMSE
    
    Scores_DT_r2.append(R2_DT)
    Scores_DT_mse.append(mse_DT)
    Scores_DT_rmse.append(rmse_DT)
    
    """SVR & NuSVR models"""
    #SVR
    Mdl_SVR = SVR(kernel ='rbf',gamma='scale',C=0.1,coef0=0.5,tol=1e-3,epsilon=0.1,cache_size=200,max_iter=-1).fit(X_train, Y_train)
    
    R2_SVR = r2_score(Y_test2, Mdl_SVR.predict(X_test2)) #r2 score
    mse_SVR = mean_squared_error(Y_test2, Mdl_SVR.predict(X_test2)) #MSE
    rmse_SVR = mean_squared_error(Y_test2, Mdl_SVR.predict(X_test2), squared=False) #RMSE
    
    Scores_SVR_r2.append(R2_SVR)
    Scores_SVR_mse.append(mse_SVR)
    Scores_SVR_rmse.append(rmse_SVR)
    
    #Nu SVR
    Mdl_NuSVR = NuSVR(nu=0.35,C=0.1,kernel='rbf',tol=1e-3,max_iter=-1).fit(X_train, Y_train)
    
    R2_NuSVR = r2_score(Y_test2, Mdl_NuSVR.predict(X_test2)) #r2 score
    mse_NuSVR = mean_squared_error(Y_test2, Mdl_NuSVR.predict(X_test2)) #MSE
    rmse_NuSVR = mean_squared_error(Y_test2, Mdl_NuSVR.predict(X_test2), squared=False) #RMSE
    
    Scores_NuSVR_r2.append(R2_NuSVR)
    Scores_NuSVR_mse.append(mse_NuSVR)
    Scores_NuSVR_rmse.append(rmse_NuSVR)

    """Neural Network (MLP) Regressor"""
    #Multi-Layer-Perceptron Neural Network
    Mdl_MLP_NN = MLPRegressor(hidden_layer_sizes=100,activation='relu',solver='adam',alpha=1e-4,batch_size='auto',learning_rate='adaptive',max_iter=200,random_state=0,tol=1e-4,epsilon=1e-10,n_iter_no_change=20).fit(X_train,Y_train)
    
    R2_MLP_NN = r2_score(Y_test2, Mdl_MLP_NN.predict(X_test2)) #r2 score
    mse_MLP_NN = mean_squared_error(Y_test2, Mdl_MLP_NN.predict(X_test2)) #MSE
    rmse_MLP_NN = mean_squared_error(Y_test2, Mdl_MLP_NN.predict(X_test2), squared=False) #RMSE

    Scores_MLP_NN_r2.append(R2_MLP_NN)
    Scores_MLP_NN_mse.append(mse_MLP_NN)
    Scores_MLP_NN_rmse.append(rmse_MLP_NN)
    
    """Nearest Neighbors Regressor"""
    #K Nearest Neighbors Mdl_KNN = KNeighborsRegressor(n_neighbors=75,weights='distance',algorithm='auto',leaf_size=30,p=2).fit(X_train, Y_train)
    Mdl_KNN = KNeighborsRegressor(n_neighbors=75,weights='distance',algorithm='auto',leaf_size=30,p=2).fit(X_train, Y_train)
    
    R2_KNN = r2_score(Y_test2, Mdl_KNN.predict(X_test2)) #r2 score
    mse_KNN = mean_squared_error(Y_test2, Mdl_KNN.predict(X_test2)) #MSE
    rmse_KNN = mean_squared_error(Y_test2, Mdl_KNN.predict(X_test2), squared=False) #RMSE
    
    Scores_KNeighbor_r2.append(R2_KNN)
    Scores_KNeighbor_mse.append(mse_KNN)
    Scores_KNeighbor_rmse.append(rmse_KNN)
 
    """Bagging (Ensemble) Regressor"""
    #Consider bagging both Trees (RF & DT)
    Mdl_Bagging = BaggingRegressor(RandomForestRegressor(),n_estimators=8,max_samples=1.0,max_features=1.0,random_state=0).fit(X_train,Y_train)
    
    R2_Bagging = r2_score(Y_test2, Mdl_Bagging.predict(X_test2)) #r2 score
    mse_Bagging = mean_squared_error(Y_test2, Mdl_Bagging.predict(X_test2)) #MSE
    rmse_Bagging = mean_squared_error(Y_test2, Mdl_Bagging.predict(X_test2), squared=False) #RMSE
    
    Scores_Bagging_r2.append(R2_Bagging)
    Scores_Bagging_mse.append(mse_Bagging)
    Scores_Bagging_rmse.append(rmse_Bagging)
    
    """Sparse Learning"""
    #ElasticNet
    Mdl_ElasticNet = ElasticNet(alpha=0.007,l1_ratio=1e-5,fit_intercept=True,max_iter=10000,tol=1e-4,warm_start=False,random_state=i,selection='random').fit(X_train,Y_train)
    
    R2_ElasticNet = r2_score(Y_test2, Mdl_ElasticNet.predict(X_test2)) #r2 score
    mse_ElasticNet = mean_squared_error(Y_test2, Mdl_ElasticNet.predict(X_test2)) #MSE
    rmse_ElasticNet = mean_squared_error(Y_test2, Mdl_ElasticNet.predict(X_test2), squared=False) #RMSE
    
    Scores_ElasticNet_r2.append(R2_ElasticNet)
    Scores_ElasticNet_mse.append(mse_ElasticNet)
    Scores_ElasticNet_rmse.append(rmse_ElasticNet)
    Weights_ElasticNet = Mdl_ElasticNet.coef_
    
    print('-- -- Iteration {} Complete -- --\n'.format(k+1))
   
    
#Notifies which city the training & testing occured at
print('Trained model on {} data, Tested on {}\n'.format(City_dict[train_idx],City_dict[test_idx]))    


"""Printing the R2 Scores, Mean, & Standard Deviation of each Model"""
print('\n-- -- The R2 Scores are -- --')
print('LinReg: r2, mean, std', hf.hf.mustd(Scores_LinRegr_r2))
print('Ridge: r2, mean, std', hf.hf.mustd(Scores_Ridge_r2))
print('SGDregr: r2, mean, std', hf.mustd(Scores_SGDregr_r2))
print('PLSregr: r2, mean, std', hf.mustd(Scores_PLSr_r2))
print('KernelRidge: r2, mean, std', hf.mustd(Scores_KernelRidge_r2))
print('RandForest: r2, mean, std', hf.mustd(Scores_RF_r2))
print('Dec.Tree: r2, mean, std', hf.mustd(Scores_DT_r2))
print('SVR: r2, mean, std', hf.mustd(Scores_SVR_r2))
print('NuSVR: r2, mean, std', hf.mustd(Scores_NuSVR_r2))
print('MLP_NN: r2, mean, std', hf.mustd(Scores_MLP_NN_r2))
print('KNeighbor: r2, mean, std', hf.mustd(Scores_KNeighbor_r2))
print('Bagging: r2, mean, std', hf.mustd(Scores_Bagging_r2))
print('ElasticNet: r2, mean, std', hf.mustd(Scores_ElasticNet_r2))

"""Printing the MSE, Mean, & Standard Deviation of each Model"""
print('\n-- -- The MSE Values are -- --')
print('LinReg: MSE, mean, std', hf.mustd(Scores_LinRegr_mse))
print('Ridge: MSE, mean, std', hf.mustd(Scores_Ridge_mse))
print('SGDregr: MSE, mean, std', hf.mustd(Scores_SGDregr_mse))
print('PLSregr: MSE, mean, std', hf.mustd(Scores_PLSr_mse))
print('KernelRidge: MSE, mean, std', hf.mustd(Scores_KernelRidge_mse))
print('RandForest: MSE, mean, std', hf.mustd(Scores_RF_mse))
print('Dec.Tree: MSE, mean, std', hf.mustd(Scores_DT_mse))
print('SVR: MSE, mean, std', hf.mustd(Scores_SVR_mse))
print('NuSVR: MSE, mean, std', hf.mustd(Scores_NuSVR_mse))
print('MLP_NN: MSE, mean, std', hf.mustd(Scores_MLP_NN_mse))
print('KNeighbor: MSE, mean, std', hf.mustd(Scores_KNeighbor_mse))
print('Bagging: MSE, mean, std', hf.mustd(Scores_Bagging_mse))
print('ElasticNet: MSE, mean, std', hf.mustd(Scores_ElasticNet_mse))

"""Printing the RMSE, Mean, & Standard Deviation of each Model"""
print('\n-- -- The RMSE Values are -- --')
print('LinReg: RMSE, mean, std', hf.mustd(Scores_LinRegr_rmse))
print('Ridge: RMSE, mean, std', hf.mustd(Scores_Ridge_rmse))
print('SGDregr: RMSE, mean, std', hf.mustd(Scores_SGDregr_rmse))
print('PLSregr: RMSE, mean, std', hf.mustd(Scores_PLSr_rmse))
print('KernelRidge: RMSE, mean, std', hf.mustd(Scores_KernelRidge_rmse))
print('RandForest: RMSE, mean, std', hf.mustd(Scores_RF_rmse))
print('Dec.Tree: RMSE, mean, std', hf.mustd(Scores_DT_rmse))
print('SVR: RMSE, mean, std', hf.mustd(Scores_SVR_rmse))
print('NuSVR: RMSE, mean, std', hf.mustd(Scores_NuSVR_rmse))
print('MLP_NN: RMSE, mean, std', hf.mustd(Scores_MLP_NN_rmse))
print('KNeighbor: RMSE, mean, std', hf.mustd(Scores_KNeighbor_rmse))
print('Bagging: RMSE, mean, std', hf.mustd(Scores_Bagging_rmse))
print('ElasticNet: RMSE, mean, std', hf.mustd(Scores_ElasticNet_rmse))


"""Creating a Dataframe for the Eeights Generated by the Models"""
#Initializing the Dataframe to hold the weights from ElasticNet
df_Model_Weights = pd.DataFrame(columns=['Year','Month','Day','Hour','Minute','DHI','DNI','GHI','Clearsky DHI','Clearsky DNI','Clearsky GHI','Cloud Type','Dew Point',
                                            'Solar Zenith Angle','Surface Albedo','Wind Speed','Precipitable Water','Wind Direction','Relative Humidity','Temperature','Pressure'])
column_names = ['Year','Month','Day','Hour','Minute','DHI','DNI','GHI','Clearsky DHI','Clearsky DNI','Clearsky GHI','Cloud Type','Dew Point',
                                            'Solar Zenith Angle','Surface Albedo','Wind Speed','Precipitable Water','Wind Direction','Relative Humidity','Temperature','Pressure']
#Creates a dataframe for the weights determined by ElasticNet
df_Model_Weights = pd.concat([df_Model_Weights, pd.DataFrame(data=np.array([Mdl_LinRegr.coef_]),columns=column_names)], ignore_index=True).dropna() #Linear Regression
df_Model_Weights = pd.concat([df_Model_Weights, pd.DataFrame(data=np.array([Mdl_Ridge.coef_]),columns=column_names)], ignore_index=True).dropna() #Ridge
df_Model_Weights = pd.concat([df_Model_Weights, pd.DataFrame(data=np.array([Mdl_SGDregr.coef_]),columns=column_names)], ignore_index=True).dropna() #Stochastic Gradient Descent Regression
df_Model_Weights = pd.concat([df_Model_Weights, pd.DataFrame(data=np.array([Mdl_ElasticNet.coef_]),columns=column_names)], ignore_index=True).dropna() #ElasticNet

