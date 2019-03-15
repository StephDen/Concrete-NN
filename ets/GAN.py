#%%
import sys

import numpy as np
import pandas as pd
import random
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from IPython.core.debugger import Tracer

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt
plt.switch_backend('agg')
#%%
from keras import backened as K
#%%
class GAN(object):
    
    def __init__(self, height= 10, channels = 1):

        self.SHAPE = (height,channels)
        self.HEIGHT = height
        self.CHANNELS = channels

        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.noise_gen = np.random.normal(0,1,(100,))

        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])

        self.stacked_G_D = self.stacked_G_D()

        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

    def generator(self):

        model = Sequential()
        #input layer
        model.add(Dense(20, input_shape = (self.HEIGHT,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        #layer 2
        model.add(Dense(40))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        #layer 3
        model.add(Dense(80))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(10, activation='tanh'))

        model.add(Reshape(self.SHAPE))
        
        return model

    def discriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense((self.HEIGHT * self.CHANNELS), input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense((self.HEIGHT * self.CHANNELS)/2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def stacked_G_D(self):
        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model
        
    def train(self, data, epochs = 2000,batch = 6, sparse_clean_ratio = 0.3):
        
        for epoch in range(epochs):
            
            # train discriminator
            random_index = np.random.randint(0, len(data) - batch//2)
            clean_mixes = data[random_index : random_index + batch//2]

            # create sparse mixes
            sparse_mix = []
            for mix in clean_mixes:
                rand_indexs = np.random.randint(0,len(mix),round(len(mix)*sparse_clean_ratio))
                mix[rand_indexs] = 0
                np.insert(sparse_mix,mix)
            
            synthetic_mixes = self.G.predict(sparse_mix)

            combined_batch = np.concatenate((clean_mixes,synthetic_mixes))
            mask_batch = np.concatenate((np.ones((batch/2, 1)), np.zeros((batch/2, 1))))

            d_loss = self.D.train_on_batch(combined_batch, mask_batch)

            # train generator
            random_index = np.random.randint(0, len(data), size = (1,batch))
            sparse_mix = []
            for mix in data[random_index]:
                rand_indexs = np.random.randint(0,len(mix),round(len(mix)*sparse_clean_ratio))
                mix[rand_indexs] = 0
                np.insert(sparse_mix,mix)
            y_label = np.ones((batch,1))

            g_loss = self.stacked_G_D.train_on_batch(noise,y_label)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
#%%
if __name__ == '__main__':

    # set random seed
    random.seed(1)
    
    
    # Load data from CSV file
    data = pd.read_csv("~/Desktop/Concrete-NN/data/ets.csv", dtype=float)
    # Removing cost for now as it is not important
    data = data.drop('Cost ($/m3)',1)
    # Creating Scaler
    scaler = MinMaxScaler(feature_range = (-1,1))
    # scaling datas
    scaled_data = scaler.fit_transform(data)
    
    gan = GAN()
    gan.train(scaled_data)

#%%
