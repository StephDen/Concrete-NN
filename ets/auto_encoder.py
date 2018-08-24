# -*- coding: utf-8 -*-
#%% Importing required libraries
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#%% Data Preprocessing
# Load data from CSV file
data = pd.read_csv("ets.csv", dtype=float)
# Removing cost for now as it is not important
data = data.drop('Cost ($/m3)',1)
# Creating Scaler
scaler = MinMaxScaler(feature_range = (0,1))
# scaling data
scaled_data = scaler.fit_transform(data)
# split data into training and test set
train_data, test_data = train_test_split(scaled_data,test_size = 0.1)

#%% Making Data Sparse
# In order to solve the overcomplete hidden layer problem we will make our
# data sparse by introducing 0s

# setting random seed
random.seed(1)
# variable for number of removed
num_rm = random.int(1,6)