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

##############################################################################
# Data loading 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/singlenpy/'
outputPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/V3'

trainIds = pd.read_csv(outputPath+'/samples/trainids.csv')
testIds = pd.read_csv(outputPath+'/samples/testmini_ids.csv')

ids = testIds['ids'].tolist() #+ trainIds['ids'].tolist()
dataSamps = np.array([basePath+'/'+val+'.npy' for val in ids])

batchSize = 16
dataSequence = DataSequence(dataSamps,batchSize,Shuffle=False)

##############################################################################
# Data loading 
###############################################################################

ModelPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/V3/encoder.h5'
Model00 = load_model(ModelPath)

EncodedData = Model00.predict(dataSequence)

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

MetaData['dim0'] = EncodedData[2][:,0]
MetaData['dim1'] = EncodedData[2][:,1]

#MetaData.to_csv(outputPath+'/metadata.csv')
