# -*- coding: utf-8 -*-
#%% Importing required libraries
import numpy as np
import pandas as pd
import random
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#%% Data Preprocessing
# Load data from CSV file
data = pd.read_csv("~/Desktop/Concrete-NN/data/ets.csv", dtype=float)
# Removing cost for now as it is not important
data = data.drop('Cost ($/m3)',1)
# Creating Scaler
scaler = MinMaxScaler(feature_range = (0,1))
# scaling data
scaled_data = scaler.fit_transform(data)
# split data into training and test set
#train_data, test_data = train_test_split(scaled_data,test_size = 0.1)

#%% Making Data Sparse
# In order to solve the overcomplete hidden layer problem we will make our
# data sparse by introducing 0s

# setting random seed
random.seed(1)
# new np array to contain 
sparse_data = np.empty((0,11))

# constructing new data set
for row_num, row in enumerate(scaled_data):
    # generate a randum number between 1 & 4 representing 
    ran_num = random.randint(1,6)
    # generate all possible combinations of indexes to be replaced
    for combination in combinations(range(10),ran_num):
        # Creating temp sparse row with a new column for row number of original
        sparse_row = np.append(row,row_num).reshape((1,11))
        # Creating sparsity
        for index in combination:
            sparse_row[0][index] = 0
        # Appending temp sparse row to np array
        sparse_data = np.append(sparse_data,sparse_row, axis=0)
#%% Splitting data into training and test data
# training
sparse_train, sparse_test = train_test_split(sparse_data, test_size = 0.2, random_state = 1)