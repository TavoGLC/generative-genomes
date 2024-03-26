#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 01:25:47 2024

@author: tavo
"""

import os
import time
import numpy as np
import pandas as pd 

import multiprocessing as mp

###############################################################################
# Blocks
###############################################################################

train = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr05/data/V1/folds/trainids.csv')
test = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr05/data/V1/folds/testids.csv')
validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr05/data/V1/folds/validationids.csv')

cont = train['ids'].tolist() + test['ids'].tolist() + validation['ids'].tolist()
cont = list(set(cont))

files = [val+'.txt' for val in cont]

###############################################################################
# Blocks
###############################################################################

MaxCPUCount=int(0.80*mp.cpu_count())

Alphabet = ['A','C','T','G']

TokenDictionary = {}

for k,val in enumerate(Alphabet):
    currentVec = [0 for j in range(len(Alphabet)+1)]
    currentVec[k] = 1
    TokenDictionary[val]=currentVec

###############################################################################
# Sequence K-mer generating functions
###############################################################################

def ReadFile(path):    
    with open(path) as f:
        lines = f.readlines()
    return str(lines[0])

###############################################################################
# Blocks
###############################################################################

def MakeSequenceEncoding(Sequence):
    
    stringFrags = [val for val in Sequence]
    nToAdd = (32*32*32) - len(stringFrags)
    toAdd = [[0,0,0,0,1] for k in range(nToAdd)]
    encoded = [TokenDictionary[val] for val in stringFrags] + toAdd    
    encoded = np.array(encoded)
    
    return encoded.astype(np.int8)

###############################################################################
# Blocks
###############################################################################

def GetDataParallel(DataBase,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [(val )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Sequence Graphs Functions
###############################################################################

matrixData = '/media/tavo/storage/biologicalSequences/covidsr04/data/singlenpy/'

seqsdir = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/archive/single/'

paths = [seqsdir+val for val in files]

chunkSize = 20000
blocksContainer = []
names = []
counter = 0

for k in range(0,len(paths),chunkSize):
    
    st = time.time()
    currentPaths = paths[k:k+chunkSize]
    names = [val[len(seqsdir):-4] for val in currentPaths]
    loadedSeqs = [ReadFile(sal) for sal in currentPaths]
    
    data = GetDataParallel(loadedSeqs,MakeSequenceEncoding)
    
    counter = counter + len(currentPaths)
    
    message = 'Sequences = ' + str(counter) + ' Time = ' + str(round((1/60)*(time.time()-st),3))
    print(message)    
    
    for nme, db in zip(names,data):
        np.save(matrixData+nme, db)
