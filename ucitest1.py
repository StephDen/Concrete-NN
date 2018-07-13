# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 22:24:21 2018

@author: Stephen
"""
#%% Loading data
import numpy as np
import pandas as pd

dataset = pd.read_csv('~/Desktop/Concrete-NN/data/Concrete_Data.csv')
dataset = dataset.iloc[:,:].values
#%% Preprocessing
# split into dep and indep
indep = dataset[:,0:8]
dep = dataset[:,8]

# feature scaling
from sklearn.preprocessing import MinMaxScaler

scalerx = MinMaxScaler()
scalery = MinMaxScaler()
indep = scalerx.fit_transform(indep)
dep = scalery.fit_transform(dep.reshape(-1,1))

#%% Building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initializing ANN
model = Sequential()

# Input layer + first layer
model.add(Dense(units = 20, activation = 'relu', input_dim = 8))

# Second layer
model.add(Dense(units = 20,activation = 'relu'))

#Third Layer
model.add(Dense(units = 12,activation = 'relu'))

# Output layer
model.add(Dense(units = 1,activation='sigmoid'))

# Compiling ANN
model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mse'])


#%% fitting and saving model
model.fit(indep,dep, epochs = 100, batch_size = 20, validation_split = 0.05)
model.save('test1.h5')
#%%
x_predict = indep[30,:].reshape(1,8)
print(scalery.inverse_transform(model.predict(x_predict)))

