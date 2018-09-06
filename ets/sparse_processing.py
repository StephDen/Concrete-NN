# -*- coding: utf-8 -*-
#%% Importing required libraries for preprocessing
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

# releasing uneeded variables to clean up variable space
del index, ran_num, row_num, sparse_row, combination, row, data, scaled_data

#%% Splitting data into training and test data

# training
sparse_train, sparse_test = train_test_split(sparse_data, test_size = 0.2, random_state = 1)

#%% Importing required libraries for neural net creation
import tensorflow as tf

#%% Creating Model

# Define model parameters

# Training
RUN_NAME = "histogram_visualization"
LEARN_RATE = 0.001
EPOCHS = 100

# Input and Output
NUM_INPUTS = 10
NUM_OUTPUTS = 10

# Hidden Layers
LAYER_1 = 15
LAYER_2 = 15
LAYER_3 = 15

# Constructing network architecture:

# weight initializer: xavier
# bias initializer: zeros
# activiation function: relu

# Input Layer
with tf.variable_scope('input'):
    input_tensor = tf.placeholder(tf.float32, shape = (None, NUM_INPUTS), name = "X")

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape = [NUM_INPUTS, LAYER_1], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = "biases1", shape = [LAYER_1], initializer = tf.zeros_initializer() )
    layer_1_output = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[LAYER_1, LAYER_2], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[LAYER_2], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[LAYER_2, LAYER_3], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[LAYER_3], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[LAYER_3, NUM_OUTPUTS], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[NUM_OUTPUTS], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases
    

# Defining the cost function of network:
with tf.variable_scope('cost'):
    output = tf.placeholder(tf.float32, shape=(None,NUM_OUTPUTS), name = "output")
    cost = tf.reduce_mean(tf.squared_difference(prediction,output))

#Defining the optimizer function that will run
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)
    
#%% Training & Logging model

# Creating summary of network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    tf.summary.histogram('predicted_value', prediction)
    summary = tf.summary.merge_all()
    
# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:
    
