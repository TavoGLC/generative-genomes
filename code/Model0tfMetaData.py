#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:05:22 2024

@author: tavo
"""

##############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf 

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K 
from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, BatchNormalization

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


ids = trainIds['ids'].tolist() + testIds['ids'].tolist()
dataSamps = np.array([basePath+'/'+val+'.npy' for val in ids])

batchSize = 16
dataSequence = DataSequence(dataSamps,batchSize,Shuffle=False)

##############################################################################
# Data loading 
###############################################################################

ModelPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/V1/AE.h5'
Model00 = load_model(ModelPath,
                     custom_objects={'KLDivergenceLayer': KLDivergenceLayer,
                                     'Sampling': Sampling,
                                     'MeanHammingDistance':MeanHammingDistance,
                                     'MinHammingDistance':MinHammingDistance})

EncoderModel = Model00.layers[1]
EncodedData = EncoderModel.predict(dataSequence)

##############################################################################
# Data loading 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')

MetaData = MetaData.loc[ids]

MetaData['date'] = pd.to_datetime(MetaData['date'])
MetaData['week'] = MetaData['date'].dt.isocalendar().week.astype(np.float32)
MetaData['dayofweek'] = MetaData['date'].dt.dayofweek.astype(np.float32)
MetaData['month'] = MetaData['date'].dt.month.astype(np.float32)
MetaData['year'] = MetaData['date'].dt.year.astype(np.float32)

MetaData['dim0'] = EncodedData[:,0]
MetaData['dim1'] = EncodedData[:,1]

##############################################################################
# Data loading 
###############################################################################

solarwnd0 = pd.read_csv('/media/tavo/storage/solar/faraday_rec.csv')
solarwnd0['date'] = pd.to_datetime(solarwnd0['date'])
solarwnd0 = solarwnd0.groupby('date').mean().rolling(3*4*7*24).mean().reset_index()
wnddaily0 = solarwnd0.groupby(pd.Grouper(freq='D', key='date')).mean()

solarwnd1 = pd.read_csv('/media/tavo/storage/solar/mag_rec.csv')
solarwnd1['date'] = pd.to_datetime(solarwnd1['date'])
solarwnd1 = solarwnd1.groupby('date').mean().rolling(3*4*7*24).mean().reset_index()
wnddaily1 = solarwnd1.groupby(pd.Grouper(freq='D', key='date')).mean()

MetaData['proton_temperature'] = wnddaily0['proton_temperature'].loc[MetaData['date']].values
MetaData['proton_speed'] = wnddaily0['proton_speed'].loc[MetaData['date']].values
MetaData['proton_density'] = wnddaily0['proton_density'].loc[MetaData['date']].values

MetaData['bt'] = wnddaily1['bt'].loc[MetaData['date']].values
MetaData['theta_gse'] = wnddaily1['theta_gse'].loc[MetaData['date']].values
MetaData['phi_gse'] = wnddaily1['phi_gse'].loc[MetaData['date']].values

feats = ['month','week','dayofyear','sunspots','daylength','bt']

target = ['dim0','dim1']

###############################################################################
# Loading packages 
###############################################################################

def ToLearnedModel():
    
    InputFunction = Input(shape=(len(feats),))
    
    X = Dense(256,use_bias=False)(InputFunction)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(128,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(64,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(8,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    Output = Dense(2)(X)
    #X = BatchNormalization()(X)
    #Output = Activation('sigmoid')(X)
    
    outModel = Model(inputs=InputFunction,outputs=Output)

    return outModel

batchSize = 64

stepsperepoch = (trainIds.shape[0])//batchSize

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001,
                                                                  stepsperepoch//2,
                                                                  t_mul=1.5,
                                                                  m_mul=0.45,
                                                                  alpha=0.01)

model = ToLearnedModel()
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn), 
                      loss='mean_absolute_percentage_error')

###############################################################################
# Loading packages 
###############################################################################

xtrain = MetaData[feats].loc[trainIds['ids'].values]
ytrain = MetaData[target].loc[trainIds['ids'].values]

xtest = MetaData[feats].loc[testIds['ids'].values]
ytest = MetaData[target].loc[testIds['ids'].values]

model.fit(xtrain, ytrain,
          validation_data=(xtest, ytest), 
          batch_size=batchSize, 
          epochs=50)

model.save(outputPath+'/converter.h5')
