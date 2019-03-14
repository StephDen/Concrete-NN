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

scalerx = MinMaxScaler((0,1))
scalery = MinMaxScaler((0,1))
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
model.add(Dense(units = 20, kernel_initializer="uniform", activation = 'relu', input_dim = 8))
model.add(Dropout(rate = 0.1))
# Second layer
model.add(Dense(units = 20,kernel_initializer="uniform",activation = 'relu'))
model.add(Dropout(rate = 0.1))
#Third Layer
model.add(Dense(units = 20,kernel_initializer="uniform",activation = 'relu'))
model.add(Dropout(rate = 0.1))

# Output layer
model.add(Dense(units = 1,kernel_initializer="uniform",activation = 'linear'))

# Compiling ANN
model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mse'])


#%% fitting and saving model
from keras.utils import plot_model
model.fit(indep,dep, epochs = 150, batch_size = 50, validation_split = 0.02)
model.save('test1.h5')
plot_model(model,to_file='model.png')
#%% loading model
from keras.models import load_model

model = load_model('test1.h5')
#%% making a prediction
x_predict = model.predict(indep[127,:].reshape(1,8))
print(scalery.inverse_transform(x_predict))

