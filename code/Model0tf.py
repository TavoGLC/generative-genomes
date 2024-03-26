#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:26:11 2024

@author: tavo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Layer
from tensorflow.keras.layers import Flatten, Reshape, BatchNormalization

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
    
    def __data_generation(self, pathList):
        
        X = []
        for k,pth in enumerate(pathList):
            encoded = np.load(pth)
            X.append(encoded)
        
        X = np.stack(X)
        Y = X

        return X,Y
    
    def __getitem__(self, index):

        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        batchIds = [self.dirList[k] for k in indexes]
        X, Y = self.__data_generation(batchIds)
        
        return X, Y

###############################################################################
# Visualization functions
###############################################################################

class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    def __init__(self,*args,**kwargs):
        self.is_placeholder=True
        super(KLDivergenceLayer,self).__init__(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*(10**-4)*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        loss = K.mean(klbatch)
        self.add_loss(loss)
        
        return inputs

class Sampling(Layer):
    '''
    Custom sampling layer
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}
    
    @tf.autograph.experimental.do_not_convert   
    def call(self,inputs,**kwargs):
        
        Mu,LogSigma=inputs
        batch=tf.shape(Mu)[0]
        dim=tf.shape(Mu)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return Mu+(K.exp(0.5*LogSigma))*epsilon
    
###############################################################################
# Loading packages 
###############################################################################

def EncoderModel(units=24):
    
    InputFunction = Input(shape=(32768,5))
    
    X = Reshape((32,32,32,5))(InputFunction)
    
    for _ in range(4):
        for _ in range(2):
            X = Conv3D(units,(3,3,3),padding='SAME')(X)
            #X = BatchNormalization()(X)
            X = Activation('selu')(X)
    
        X = Conv3D(units,(3,3,3),strides=(1,1,2),padding='SAME')(X)
        #X = BatchNormalization()(X)
        X = Activation('selu')(X)
    
    X = Reshape((32,32,2*units))(X)
    
    for _ in range(4):
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
    Mu,LogSigma = KLDivergenceLayer()([Mu,LogSigma])
    Output = Sampling()([Mu,LogSigma])

    encoder = Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction, encoder

def DecoderModel(units=24):
    
    InputFunction = Input(shape=(2,))
    
    X = Dense(16,use_bias=False)(InputFunction)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    X = Dense(64,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    X = Dense(32*2*units,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('selu')(X)
    
    X = Reshape((32,2,units))(X)
    
    for _ in range(4):
        for _ in range(2):
            X = Conv2D(units,(3,3),padding='SAME',use_bias=False)(X)
            X = BatchNormalization()(X)
            X = Activation('selu')(X)
    
        X = Conv2DTranspose(units,(3,3),strides=(1,2),padding='SAME',use_bias=False)(X)
        X = BatchNormalization()(X)
        X = Activation('selu')(X)
    
    X = Reshape((32,32,2,units//2))(X)
    
    for _ in range(4):
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
    
    output = Reshape((32768,5))(X)
    
    decoder = Model(inputs=InputFunction,outputs=output)
    
    return InputFunction, decoder

def VAEModel(units=24):
    
    InputEncoder,Encoder = EncoderModel(units=units)
    InputDecoder,Decoder = DecoderModel(units=units)
    Output = Decoder(Encoder(InputEncoder))
    VAE = Model(inputs=InputEncoder,outputs=Output)
    
    return VAE

##############################################################################
# Data loading 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/singlenpy/'
outputPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/V1'

#trainIds = pd.read_csv(outputPath+'/folds/trainmini_ids.csv')
#testIds = pd.read_csv(outputPath+'/folds/testmini_ids.csv')

#trainIds = pd.read_csv(outputPath+'/folds/trainmid_ids.csv')
#testIds = pd.read_csv(outputPath+'/folds/testmid_ids.csv')

trainIds = pd.read_csv(outputPath+'/folds/trainids.csv')
testIds = pd.read_csv(outputPath+'/folds/testids.csv')

###############################################################################
# Loading packages 
###############################################################################

batchSize = 16

np.random.seed(128)
trainLabels = trainIds['ids'].values
testLabels = testIds['ids'].values

#trainLabels = trainLabels[0:50*batchSize] 
#testLabels = testLabels[0:10*batchSize]

trainSamps = np.array([basePath+'/'+val+'.npy' for val in trainLabels])
testSamps = np.array([basePath+'/'+val+'.npy' for val in testLabels])

trainSequence = DataSequence(trainSamps,batchSize)
testSequence = DataSequence(testSamps,batchSize,Shuffle=False)

###############################################################################
# Loading packages 
###############################################################################

epochs = 10

stepsperepoch = (trainSamps.shape[0])//batchSize
steps = epochs*stepsperepoch

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001,
                                                                  stepsperepoch//2,
                                                                  t_mul=1.5,
                                                                  m_mul=0.45,
                                                                  alpha=0.2)

###############################################################################
# Loading packages 
###############################################################################

def HammingDistanceBatch(y_true,y_pred):
    
    ytrue = tf.math.argmax(y_true,axis=-1)
    ypred = tf.math.argmax(y_pred,axis=-1)
    
    difference = tf.math.not_equal(ytrue,ypred)
    difference = tf.cast(difference,dtype=tf.float32)
    
    dist = tf.math.reduce_sum(difference,axis=-1)
    
    return dist

def MeanHammingDistance(y_true,y_pred):
    dist = HammingDistanceBatch(y_true, y_pred)
    return tf.reduce_mean(dist)

def MinHammingDistance(y_true,y_pred):
    dist = HammingDistanceBatch(y_true, y_pred)
    return tf.reduce_min(dist)

def Loss(y_true,y_pred):
    
    alpha = 0.05
    
    ytr0 = tf.reduce_sum(y_true[:,:,0:-1],axis=-1)
    ypr0 = tf.reduce_sum(y_pred[:,:,0:-1],axis=-1)
    
    ytr1 = y_true[:,:,-1]
    ypr1 = y_pred[:,:,-1]
    
    ybtr = tf.stack([ytr0,ytr1],axis=-1)
    ybpr = tf.stack([ypr0,ypr1],axis=-1)
    
    l1 = (1-alpha)*tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    l2 = alpha*tf.keras.losses.binary_crossentropy(ybtr,ybpr)
    
    loss = l1+l2
    
    return loss

###############################################################################
# Loading packages 
###############################################################################

model = VAEModel(units=14)
model.summary()

model.compile(optimizer=Adam(learning_rate=lr_decayed_fn), 
              loss=Loss,
              metrics=[MeanHammingDistance,MinHammingDistance])

model.fit(trainSequence,validation_data=testSequence, 
          batch_size=batchSize, epochs=5)

model.layers[2].save(outputPath+'/test.h5')

model.save(outputPath+'/AE.h5')


