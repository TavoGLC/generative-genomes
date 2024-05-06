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
import matplotlib.colors as colors

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

Model = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00infneu/model/decoder.h5')

samples = 150

x = np.linspace(-4, 10, samples)
y = np.linspace(-9, 6, samples)

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

data0 = []
data1 = []
data2 = []
data3 = []

for batch in container:
    decoded_seqs = Model.predict(batch,verbose=0).argmax(axis=-1)
    batch_vals = batch.numpy()
    
    for k,sq in enumerate(decoded_seqs):
        
        local_gene = ''.join([toLetters[sal] for sal in sq])
        valid_gene = ''.join([i for i in local_gene if i in valid])[0:900]
        
        currentSeq = Seq(valid_gene)
        translatedSeq = str(currentSeq.translate()).replace('*','')
        
        reg0 = translatedSeq[115:120]
        reg1 = translatedSeq[150:156]
        reg2 = translatedSeq[175:180]
        reg3 = translatedSeq[220:230]
        
        localdata0 = []
        localdata0 = localdata0 + list(batch_vals[k])
        feats = pd.concat([adata[val] for val in reg0],axis=1).mean(axis=1)            
        localdata0 = localdata0 + feats.tolist() 
        data0.append(localdata0)
        
        localdata1 = []
        localdata1 = localdata1 + list(batch_vals[k])
        feats = pd.concat([adata[val] for val in reg1],axis=1).mean(axis=1)            
        localdata1 = localdata1 + feats.tolist() 
        data1.append(localdata1)
        
        localdata2 = []
        localdata2 = localdata2 + list(batch_vals[k])
        feats = pd.concat([adata[val] for val in reg2],axis=1).mean(axis=1)            
        localdata2 = localdata2 + feats.tolist() 
        data2.append(localdata2)
        
        localdata3 = []
        localdata3 = localdata3 + list(batch_vals[k])
        feats = pd.concat([adata[val] for val in reg3],axis=1).mean(axis=1)            
        localdata3 = localdata3 + feats.tolist() 
        data3.append(localdata3)
        
        

featnames = ['feat'+str(k) for k in range(566)] 
colnames = ['dim0','dim1'] + featnames
coldesc = adata['description'].values

data0 = np.array(data0)
data0 = pd.DataFrame(data0,columns=colnames)

data1 = np.array(data1)
data1 = pd.DataFrame(data1,columns=colnames)

data2 = np.array(data2)
data2 = pd.DataFrame(data2,columns=colnames)

data3 = np.array(data3)
data3 = pd.DataFrame(data3,columns=colnames)

##############################################################################
# Loading packages 
###############################################################################

regionRange = ['115-120','150-156','175-180','220-230']

for k,val in enumerate(featnames[0:10]):
    
    fig, axs = plt.subplots(1, 4, figsize=(20,7))
    
    maxval = np.max([sal[val].max() for sal in [data0,data1,data2,data3]])
    minval = np.min([sal[val].min() for sal in [data0,data1,data2,data3]])
    norm = colors.Normalize(vmin=minval,vmax=maxval)
    
    for j,df in enumerate([data0,data1,data2,data3]):
        
        zz = df[val].values.reshape(samples,samples,order='F')
        img = axs[j].pcolormesh(xx, yy, zz,shading='gouraud',norm=norm)
        fig.colorbar(img)
        axs[j].set_title('Neuraminidase Region ' + regionRange[j])

    fig.suptitle('Mean ' + coldesc[k] + ' of the hemagglutinin protein RBD')
