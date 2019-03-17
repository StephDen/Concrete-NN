#%%
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt
#%%
#from keras import backened as K
#%%
class GAN(object):
    
    def __init__(self, height= 10, channels = 1):

        self.SHAPE = (height,channels)
        self.HEIGHT = height
        self.CHANNELS = channels

        self.OPTIMIZER = Adam(lr=0.0002, beta_1 = 0.5,decay=8e-9)

        self.G = self._generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

        self.D = self._discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])

        self.stacked_G_D = self._stacked_G_D()

        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

        self.G_loss = []
        self.D_C_loss = []
        self.D_S_loss = []
        self.EPOCHS = 2000
    def _generator(self):

        model = Sequential()
        #input layer
        model.add(Dense(20, input_shape = (self.HEIGHT,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        #layer 2
        model.add(Dense(40))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

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

    def _discriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense((self.HEIGHT * self.CHANNELS), input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.int64((self.HEIGHT * self.CHANNELS)/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def _stacked_G_D(self):
        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model
        
    def train(self, data, epochs = 2000,batch = 6, sparse_clean_ratio = 0.3):
        self.EPOCHS = epochs
        for epoch in range(epochs):
            
            # train discriminator
            random_index = np.random.randint(0, len(data) - batch//2)
            clean_mixes = data[random_index : random_index + batch//2]

            # create sparse mixes
            sparse_mix = clean_mixes
            for mix in sparse_mix:
                rand_indexs = np.random.randint(0,len(mix),round(len(mix)*sparse_clean_ratio))
                for i in rand_indexs:
                    mix[i]=0
            
            synthetic_mixes = self.G.predict_on_batch(sparse_mix)

            # combined_batch = np.concatenate((clean_mixes.reshape(synthetic_mixes.shape),synthetic_mixes))
            # mask_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

            # d_loss = self.D.train_on_batch(combined_batch, mask_batch)
            d_loss_clean = self.D.train_on_batch(clean_mixes.reshape(synthetic_mixes.shape),np.ones((np.int64(batch/2), 1)))
            d_loss_synthetic = self.D.train_on_batch(synthetic_mixes,np.zeros((np.int64(batch/2), 1)))

            
            # train generator
            random_index = np.random.randint(0, len(data), size = (1,batch))
            sparse_mix = data[random_index]
            for mix in sparse_mix:
                rand_indexs = np.random.randint(0,high=len(mix),size=round(len(mix)*sparse_clean_ratio))
                mix[rand_indexs] = 0
            y_label = np.ones((batch,1))

            g_loss = self.stacked_G_D.train_on_batch(np.squeeze(sparse_mix),y_label)

            self.G_loss.append(g_loss)
            self.D_C_loss.append(d_loss_clean)
            self.D_S_loss.append(d_loss_synthetic)
            print ('epoch: %d, [Discriminator :: loss: %f,%f], [ Generator :: loss: %f]' % (epoch, d_loss_clean[0],d_loss_synthetic[0], g_loss))
    def plot(self):
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-whitegrid')
        
        fig = plt.figure()
        ax = plt.axes()

        x = range(self.EPOCHS)
        ax.plot(x,self.G_loss, color = 'orange')
        ax.plot(x,self.D_C_loss, 'b-', alpha = 0.5)
        ax.plot(x,self.D_S_loss, 'g-', alpha = 0.5)
        plt.show()
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
    np.random.shuffle(scaled_data)

    gan = GAN()
    gan.train(scaled_data, epochs=5000,batch = 4,sparse_clean_ratio=0.2)
    gan.plot()
#%%
    from keras.utils import plot_model
    plot_model(gan, to_file='model.png')
