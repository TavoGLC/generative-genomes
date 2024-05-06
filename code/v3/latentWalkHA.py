#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:11:03 2024

@author: tavo
"""

##############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from Bio.Seq import Seq

import tensorflow as tf
from tensorflow.keras.models import load_model

##############################################################################
# Loading packages 
###############################################################################

toLetters = {}
Alphabet = ['A','C','T','G','e']
for k,val in enumerate(Alphabet):
    toLetters[k] = val

valid = ['A','C','T','G']

##############################################################################
# Loading packages 
###############################################################################

adata = pd.read_csv('/media/tavo/storage/biologicalSequences/genomes/archive/aaindex1.csv')

Model = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00infha/model/decoder.h5')

samples = 250

x = np.linspace(-5, 10, samples)
y = np.linspace(-12, 5, samples)

xx,yy = np.meshgrid(x,y)

container = []

for xv in x:
    for yv in y:
        container.append([xv,yv])

container = np.array(container)
container = tf.data.Dataset.from_tensor_slices(container).batch(16)

##############################################################################
# Loading packages 
###############################################################################

data = []

for batch in container:
    decoded_seqs = Model.predict(batch,verbose=0).argmax(axis=-1)
    batch_vals = batch.numpy()
    
    for k,sq in enumerate(decoded_seqs):
        
        local_gene = ''.join([toLetters[sal] for sal in sq])
        valid_gene = ''.join([i for i in local_gene if i in valid])[0:900]
        
        currentSeq = Seq(valid_gene)
        translatedSeq = str(currentSeq.translate()).replace('*','')
        rbd = translatedSeq[50:290]
        localdata = []
        
        if rbd.find('*')==-1:
            localdata = localdata + list(batch_vals[k])
            feats = pd.concat([adata[val] for val in rbd],axis=1).mean(axis=1)            
            localdata = localdata + feats.tolist()   
        
        else:
            localdata = localdata + list(batch_vals[k])
            localdata = localdata + [np.nan for k in range(566)] 
        
        data.append(localdata)

data = np.array(data)
featnames = ['feat'+str(k) for k in range(566)] 
colnames = ['dim0','dim1'] + featnames
coldesc = adata['description'].values

data = pd.DataFrame(data,columns=colnames)

##############################################################################
# Loading packages 
###############################################################################

for k,val in enumerate(featnames[0:10]):
    
    fig = plt.figure(figsize=(12,10))
    ax = plt.gca()
    zz = data[val].values.reshape(samples,samples,order='F')
    img = ax.pcolormesh(xx, yy, zz,shading='gouraud')
    fig.colorbar(img)
    ax.set_title('Mean ' + coldesc[k] + ' of the spike protein RBD')
