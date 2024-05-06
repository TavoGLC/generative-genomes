#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:20:56 2024

@author: tavo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import SeqIO
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, BatchNormalization

###############################################################################
# Loading packages 
###############################################################################

outputPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/v4'
modelPath = outputPath + '/model'
plotsPath = outputPath + '/images'

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/genomes/infneu/sequences.csv')

MetaData['date'] = pd.to_datetime(MetaData['Collection_Date'],format='%Y-%m-%d',errors='coerce')
MetaData['dayofyear'] = MetaData['date'].dt.dayofyear

MetaData = MetaData.set_index('Accession')

###############################################################################
# Visualization functions
###############################################################################

seqsfile = SeqIO.parse("/media/tavo/storage/biologicalSequences/genomes/infneu/sequences.fasta", "fasta")
seqs = []
ids = []

def GetFragment(sequence):
    
    seq = str(sequence.seq)
    start = seq.find('ATG')
    stopc = ['TAA','TAG','TGA']
    stops = [1 if seq[k:k+3] in stopc else 0 for k in range(len(seq)-3)]
    ends = [k for k,val in enumerate(stops) if val==1]
    
    return seq[start:ends[-1]+3]

for record in seqsfile:
    seqs.append(GetFragment(record))
    ids.append(record.id)

Alphabet = ['A','C','T','G']

TokenDictionary = {}

for k,val in enumerate(Alphabet):
    currentVec = [0 for j in range(len(Alphabet)+1)]
    currentVec[k] = 1
    TokenDictionary[val]=currentVec

ids = np.array(ids)
MetaData = MetaData.loc[ids]

###############################################################################
# Blocks
###############################################################################

def MakeSequenceEncoding(sequence):
    
    stringFrags = [val for val in sequence]
    nToAdd = (8*16*16) - len(stringFrags)
    toAdd = [[0,0,0,0,1] for k in range(nToAdd)]
    encoded = []
    for val in stringFrags:
        if val in TokenDictionary.keys():
            encoded.append(TokenDictionary[val])
        else:
            encoded.append([0,0,0,0,1])
            
    encoded = encoded + toAdd    
    encoded = np.array(encoded)
    
    return encoded.astype(np.int8)

###############################################################################
# Blocks
###############################################################################

gdata = [MakeSequenceEncoding(val) for val in seqs]
gdata = np.stack(gdata)

indexs = np.arange(gdata.shape[0])
itrain, itest, _, _ = train_test_split(indexs, indexs, test_size=0.1, 
                                       random_state=42)

Xtrain = gdata[itrain]
Xtest = gdata[itest]

finalids = ids[itest]

MetaData = MetaData.loc[finalids]

###############################################################################
# Visualization functions
###############################################################################

globalSeed=768

tf.keras.utils.set_random_seed(globalSeed)
    
###############################################################################
# Visualization functions
###############################################################################

class DataSequence(Sequence):
    
    def __init__(self, DirList,BatchSize,Shuffle=True):
        
        self.dirList = DirList
        self.batchSize = BatchSize
        self.shuffle = Shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.dirList)/self.batchSize))
    
    def on_epoch_end(self):

        self.indexes = np.arange(len(self.dirList))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexs):
        
        X = self.dirList[indexs]

        return X
    
    def __getitem__(self, index):

        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        X = self.__data_generation(indexes)
        
        return X
    
###############################################################################
# Loading packages 
###############################################################################

@tf.keras.utils.register_keras_serializable('Sampling')
def Sampling(inputs):
    Mu,LogSigma=inputs
    batch=tf.shape(Mu)[0]
    dim=tf.shape(Mu)[1]
    epsilon=K.random_normal(shape=(batch,dim))

    return Mu+(K.exp(0.5*LogSigma))*epsilon

def EncoderModel(units=24):
    
    InputFunction = Input(shape=(2048,5))
    
    X = Reshape((8,16,16,5))(InputFunction)
    
    for _ in range(3):
        for _ in range(2):
            X = Conv3D(units,(3,3,3),padding='SAME')(X)
            #X = BatchNormalization()(X)
            X = Activation('selu')(X)
    
        X = Conv3D(units,(3,3,3),strides=(1,1,2),padding='SAME')(X)
        #X = BatchNormalization()(X)
        X = Activation('selu')(X)
    
    X = Reshape((8,16,2*units))(X)
    
    for _ in range(3):
        for _ in range(2):
            X = Conv2D(units,(3,3),padding='SAME',use_bias=False)(X)
            X = BatchNormalization()(X)
            X = Activation('selu')(X)
    
        X = Conv2D(units,(3,3),strides=(1,2),padding='SAME',use_bias=False)(X)
        X = BatchNormalization()(X)
        X = Activation('selu')(X)
    
    X = Flatten()(X)
    X = Dense(64,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    X = Dense(16,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    Mu = Dense(2)(X)
    LogSigma = Dense(2)(X)
    samples = Sampling([Mu,LogSigma])
    Output = BatchNormalization()(samples)

    encoder = Model(inputs=InputFunction,outputs=[Mu,LogSigma,Output])
    
    return encoder

def DecoderModel(units=24):
    
    InputFunction = Input(shape=(2,))
    
    X = Dense(16,use_bias=False)(InputFunction)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    X = Dense(64,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    X = Dense(8*2*units,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    X = Reshape((8,2,units))(X)
    
    for _ in range(3):
        for _ in range(2):
            X = Conv2D(units,(3,3),padding='SAME',use_bias=False)(X)
            X = BatchNormalization()(X)
            X = Activation('selu')(X)
    
        X = Conv2DTranspose(units,(3,3),strides=(1,2),padding='SAME',use_bias=False)(X)
        X = BatchNormalization()(X)
        X = Activation('selu')(X)
    
    X = Reshape((8,16,2,units//2))(X)
    
    for _ in range(3):
        for _ in range(2):    
            X = Conv3D(units,(3,3,3),padding='SAME')(X)
            #X = BatchNormalization()(X)
            X = Activation('selu')(X)
        
        X = Conv3DTranspose(units,(3,3,3),strides=(1,1,2),padding='SAME')(X)
        #X = BatchNormalization()(X)
        X = Activation('selu')(X)
    
    X = Conv3D(5,(3,3,3),padding='SAME')(X)
    #X = BatchNormalization()(X)
    X = Activation('softmax')(X)
    
    output = Reshape((2048,5))(X)
    
    decoder = Model(inputs=InputFunction,outputs=output)
    
    return decoder

###############################################################################
# Loading packages 
###############################################################################

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")        
        self.hamming_loss_tracker = tf.keras.metrics.Mean(name="hamming_loss")
        
        self.mean_hamming_distance_loss_tracker = tf.keras.metrics.Mean(name="MeanHammingDistance")
        self.max_hamming_distance_loss_tracker = tf.keras.metrics.Mean(name="MaxHammingDistance")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.hamming_loss_tracker,      
            self.mean_hamming_distance_loss_tracker,
            self.max_hamming_distance_loss_tracker
        ]
    
    def KLDivergenceLoss(self,mean,logvar):
        klscale = 10**-4
        loss = -0.5*klscale*(1 + logvar - tf.square(mean) - tf.exp(logvar))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss 
    
    def HammingDist(self,data,reconstruction):
        
        ytrue = tf.math.argmax(data,axis=-1)
        ypred = tf.math.argmax(reconstruction,axis=-1)
            
        difference = tf.math.not_equal(ytrue,ypred)
        difference = tf.cast(difference,dtype=tf.float32)
            
        dist = tf.math.reduce_sum(difference,axis=-1)
        
        return dist
    
    def TotalModelLoss(self,data,reconstruction,zmean,zlog):
            
        # Classification/reconstruction loss
        l1 = tf.keras.losses.categorical_crossentropy(data, reconstruction)
            
        # KL divergence loss
        kl_loss = self.KLDivergenceLoss(zmean, zlog)
            
        #hamminglosss
        seqlength = tf.cast(tf.shape(data)[1],tf.float32)
        hloss = tf.reduce_sum(tf.cast(data, tf.float32)*reconstruction,axis=(1,2))*(1/seqlength)
        hloss = 1-tf.reduce_mean(hloss)
        
        # Total loss
        total_loss = l1 + kl_loss + hloss
            
        return total_loss,l1,kl_loss,hloss
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            total_loss,l1,kl_loss,hloss = self.TotalModelLoss(data, reconstruction, z_mean, z_log_var)
            
            # Metrics
            dist = self.HammingDist(data, reconstruction)
            
            hdmean = tf.reduce_mean(dist)
            hdmax = tf.reduce_max(dist)
            
        #Updating the weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        #Updating the metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(l1)
        self.kl_loss_tracker.update_state(kl_loss)
        
        self.hamming_loss_tracker.update_state(hloss)

        self.mean_hamming_distance_loss_tracker.update_state(hdmean)
        self.max_hamming_distance_loss_tracker.update_state(hdmax)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "hamming_loss": self.hamming_loss_tracker.result(),
            "MeanHammingDistance": self.mean_hamming_distance_loss_tracker.result(),
            "MaxHammingDistance": self.max_hamming_distance_loss_tracker.result(),
        }
    
    def test_step(self,data):
        
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        total_loss,l1,kl_loss,hloss = self.TotalModelLoss(data, reconstruction, z_mean, z_log_var)
        
        # Metrics
        dist = self.HammingDist(data, reconstruction)
        
        hdmean = tf.reduce_mean(dist)
        hdmax = tf.reduce_max(dist)
        
        #Updating the metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(l1)
        self.kl_loss_tracker.update_state(kl_loss)
        self.hamming_loss_tracker.update_state(hloss)

        self.mean_hamming_distance_loss_tracker.update_state(hdmean)
        self.max_hamming_distance_loss_tracker.update_state(hdmax)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "hamming_loss": self.hamming_loss_tracker.result(),
            "MeanHammingDistance": self.mean_hamming_distance_loss_tracker.result(),
            "MaxHammingDistance": self.max_hamming_distance_loss_tracker.result(),
        }

##############################################################################
# Data loading 
##############################################################################

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.main_loss = []
        self.reconstruction_loss = []
        self.kl_loss = []
        self.hamming_loss = []
        self.MinHammingDistance = []
        self.MeanHammingDistance = []
        self.MaxHammingDistance = []

    def on_batch_end(self, batch, logs={}):
        self.main_loss.append(logs.get('loss'))
        self.reconstruction_loss.append(logs.get('reconstruction_loss'))
        self.kl_loss.append(logs.get('kl_loss'))
        self.hamming_loss.append(logs.get('hamming_loss'))
        self.MeanHammingDistance.append(logs.get('MeanHammingDistance'))
        self.MaxHammingDistance.append(logs.get('MaxHammingDistance'))
    
    def on_epoch_end(self, epoch, logs=None):
        
        zmean,zlogvar,z = self.model.encoder.predict(miniSequence)
        
        MetaData['zmean0'] = zmean[:,0]
        MetaData['zmean1'] = zmean[:,1]
        MetaData['zlogvar0'] = zlogvar[:,0]
        MetaData['zlogvar1'] =zlogvar[:,1]
        MetaData['dim0'] = z[:,0]
        MetaData['dim1'] = z[:,1]
        
        fig,axs = plt.subplots(1,3,figsize=(17,7))
        MetaData.plot.scatter(x='zmean0',y='zmean1',c='dayofyear',cmap='viridis',ax=axs[0])
        MetaData.plot.scatter(x='zlogvar0',y='zlogvar1',c='dayofyear',cmap='viridis',ax=axs[1])
        MetaData.plot.scatter(x='dim0',y='dim1',c='dayofyear',cmap='viridis',ax=axs[2])
        plt.tight_layout()
        plt.savefig(plotsPath + '/figlatent'+str(epoch)+'.png')
        plt.close()

batchLoss = LossHistory()

###############################################################################
# Loading packages 
###############################################################################

batchSize = 16

stepsperepoch = (Xtrain.shape[0])//batchSize

lr_decayed_fn = tf.keras.optimizers.schedules.InverseTimeDecay(0.0005,
                                                               7*stepsperepoch,
                                                               4)

###############################################################################
# Loading packages 
###############################################################################

trainSequence = DataSequence(Xtrain,batchSize)
testSequence = DataSequence(Xtest,batchSize,Shuffle=False)
miniSequence = DataSequence(Xtest,batchSize,Shuffle=False)

epochs = 15

encoder = EncoderModel(units=24)
decoder = DecoderModel(units=24)

model = VAE(encoder, decoder)
#model.summary()

model.compile(optimizer=Adam(learning_rate=lr_decayed_fn))

model.fit(trainSequence,validation_data=testSequence,
          batch_size=batchSize, epochs=epochs,
          callbacks=[batchLoss])

model.encoder.save(modelPath + '/encoder.h5')
model.decoder.save(modelPath + '/decoder.h5')

###############################################################################
# Loading packages 
###############################################################################

fig,axs = plt.subplots(6,2,figsize=(17,17))

axs[0,0].plot(model.history.history['loss'],label='loss')
axs[0,0].plot(model.history.history['val_loss'],label='val_loss')
axs[0,0].legend(loc=1)

axs[0,1].plot(batchLoss.main_loss,label='loss')
axs[0,1].legend(loc=1)
axs[0,1].set_yscale('log')

axs[1,0].plot(model.history.history['reconstruction_loss'],label='reconstruction_loss')
axs[1,0].plot(model.history.history['val_reconstruction_loss'],label='val_reconstruction_loss')
axs[1,0].legend(loc=1)

axs[1,1].plot(batchLoss.reconstruction_loss,label='reconstruction_loss')
axs[1,1].legend(loc=1)
axs[1,1].set_yscale('log')

axs[2,0].plot(model.history.history['kl_loss'],label='kl_loss')
axs[2,0].plot(model.history.history['val_kl_loss'],label='val_kl_loss')
axs[2,0].legend(loc=1)

axs[2,1].plot(batchLoss.kl_loss,label='kl_loss')
axs[2,1].legend(loc=1)
axs[2,1].set_yscale('log')

axs[3,0].plot(model.history.history['hamming_loss'],label='hamming_loss')
axs[3,0].plot(model.history.history['val_hamming_loss'],label='val_hamming_loss')
axs[3,0].legend(loc=1)

axs[3,1].plot(batchLoss.hamming_loss,label='hamming_loss')
axs[3,1].legend(loc=1)
axs[3,1].set_yscale('log')

axs[4,0].plot(model.history.history['MeanHammingDistance'],label='MeanHammingDistance')
axs[4,0].plot(model.history.history['val_MeanHammingDistance'],label='val_MeanHammingDistance')
axs[4,0].legend(loc=1)

axs[4,1].plot(batchLoss.MeanHammingDistance,label='MeanHammingDistance')
axs[4,1].legend(loc=1)
axs[4,1].set_yscale('log')

axs[5,0].plot(model.history.history['MaxHammingDistance'],label='MaxHammingDistance')
axs[5,0].plot(model.history.history['val_MaxHammingDistance'],label='val_MaxHammingDistance')
axs[5,0].legend(loc=1)

axs[5,1].plot(batchLoss.MaxHammingDistance,label='MaxHammingDistance')
axs[5,1].legend(loc=1)
axs[5,1].set_yscale('log')

plt.tight_layout()
plt.savefig(plotsPath + '/figTrainingSmall.png')
plt.close()
