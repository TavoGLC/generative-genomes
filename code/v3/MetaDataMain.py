#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:05:22 2024

@author: tavo
"""

##############################################################################
# Loading packages 
###############################################################################

import json
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from Bio import SeqIO

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

###############################################################################
# Visualization functions
###############################################################################
    
class DataSequenceFile(Sequence):
    
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

        return X
    
    def __getitem__(self, index):

        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        batchIds = [self.dirList[k] for k in indexes]
        X = self.__data_generation(batchIds)
        
        return X
    
class DataSequenceLocal(Sequence):
    
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

nSamples = 2500

def ProcessMetaData(df):
    
    df['date'] = pd.to_datetime(df['Collection_Date'],format='%Y-%m-%d',errors='coerce')
    df['dayofyear'] = df['date'].dt.dayofyear
    df = df.set_index('Accession')

    disc = df.shape[0] - df['dayofyear'].isna().sum()
    
    if disc<nSamples:            
        df = df.fillna(-1)
    else:
        df = df[df['dayofyear'].notna()]
        
    index = np.arange(df.shape[0])
    np.random.shuffle(index)
    df = df.iloc[index[0:nSamples]]
    
    return df

def GetFragment(sequence):
    
    seq = str(sequence.seq)
    start = seq.find('ATG')
    stopc = ['TAA','TAG','TGA']
    stops = [1 if seq[k:k+3] in stopc else 0 for k in range(len(seq)-3)]
    ends = [k for k,val in enumerate(stops) if val==1]
    
    return seq[start:ends[-1]+3]

def SelectGenomes(data,index):
    
    ids = []
    seqs = []
    for record in data:
        local_id = record.id
        if local_id in index:
            seqs.append(GetFragment(record))
            ids.append(record.id)
    
    return ids,seqs

Alphabet = ['A','C','T','G']

TokenDictionary = {}

for k,val in enumerate(Alphabet):
    currentVec = [0 for j in range(len(Alphabet)+1)]
    currentVec[k] = 1
    TokenDictionary[val]=currentVec

def MakeSequenceEncoding(sequence,finalsize):
    
    stringFrags = [val for val in sequence]
    nToAdd = finalsize - len(stringFrags)
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
# Loading packages 
###############################################################################

genomedir = '/media/tavo/storage/biologicalSequences/genomes'
batchSize = 16


infha_md = pd.read_csv(genomedir+'/infha/sequences.csv')
infha_md = ProcessMetaData(infha_md)

infha_seq = SeqIO.parse(genomedir + "/infha/sequences.fasta", "fasta")
infha_sel = SelectGenomes(infha_seq, infha_md.index)

infha_md = infha_md.loc[infha_sel[0]]
infha_md['seq'] =  infha_sel[1]

encinfha = [MakeSequenceEncoding(sq, 8*16*16) for sq in infha_md['seq']]
encinfha = np.stack(encinfha)


dataSequence = DataSequenceLocal(encinfha,batchSize,Shuffle=False)

Model_infha = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00infha/model/encoder.h5')

Encoded_infha = Model_infha.predict(dataSequence)

infha_md['Parameter-1'] = Encoded_infha[2][:,0]
infha_md['Parameter-2'] = Encoded_infha[2][:,1]

infha_md['virus'] = ['INFHA' for _ in range(nSamples)]

##############################################################################
# Data loading 
###############################################################################

infneu_md = pd.read_csv(genomedir+'/infneu/sequences.csv')
infneu_md = ProcessMetaData(infneu_md)

infneu_seq = SeqIO.parse(genomedir + "/infneu/sequences.fasta", "fasta")
infneu_sel = SelectGenomes(infneu_seq, infneu_md.index)

infneu_md = infneu_md.loc[infneu_sel[0]]
infneu_md['seq'] =  infneu_sel[1]

encinfneu = [MakeSequenceEncoding(sq, 8*16*16) for sq in infneu_md['seq']]
encinfneu = np.stack(encinfneu)

dataSequence = DataSequenceLocal(encinfneu,batchSize,Shuffle=False)

Model_infneu = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00infneu/model/encoder.h5')

Encoded_infneu = Model_infneu.predict(dataSequence)

infneu_md['Parameter-1'] = Encoded_infneu[2][:,0]
infneu_md['Parameter-2'] = Encoded_infneu[2][:,1]

infneu_md['virus'] = ['INFNEU' for _ in range(nSamples)]

##############################################################################
# Data loading 
###############################################################################

infm2_md = pd.read_csv(genomedir+'/infm2/sequences.csv')
infm2_md = ProcessMetaData(infm2_md)

infm2_seq = SeqIO.parse(genomedir + "/infm2/sequences.fasta", "fasta")
infm2_sel = SelectGenomes(infm2_seq, infm2_md.index)

infm2_md = infm2_md.loc[infm2_sel[0]]
infm2_md['seq'] =  infm2_sel[1]

encinfm2 = [MakeSequenceEncoding(sq, 8*16*16) for sq in infm2_md['seq']]
encinfm2 = np.stack(encinfm2)

dataSequence = DataSequenceLocal(encinfm2,batchSize,Shuffle=False)

Model_infm2 = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00infm2/model/encoder.h5')

Encoded_infm2 = Model_infm2.predict(dataSequence)

infm2_md['Parameter-1'] = Encoded_infm2[2][:,0]
infm2_md['Parameter-2'] = Encoded_infm2[2][:,1]

infm2_md['virus'] = ['INFM2' for _ in range(nSamples)]

##############################################################################
# Data loading 
###############################################################################

rota_md = pd.read_csv(genomedir+'/rota/sequences.csv')
rota_md = ProcessMetaData(rota_md)

rota_seq = SeqIO.parse(genomedir + "/rota/sequences.fasta", "fasta")
rota_sel = SelectGenomes(rota_seq, rota_md.index)

rota_md = rota_md.loc[rota_sel[0]]
rota_md['seq'] =  rota_sel[1]

encrota = [MakeSequenceEncoding(sq, 16*16*16) for sq in rota_md['seq']]
encrota = np.stack(encrota)

dataSequence = DataSequenceLocal(encrota,batchSize,Shuffle=False)

Model_rota = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00rota/model/encoder.h5')

Encoded_rota = Model_rota.predict(dataSequence)

rota_md['Parameter-1'] = Encoded_rota[2][:,0]
rota_md['Parameter-2'] = Encoded_rota[2][:,1]

rota_md['virus'] = ['ROTA' for _ in range(nSamples)]


##############################################################################
# Data loading 
###############################################################################

vhb_md = pd.read_csv(genomedir+'/vhb/sequences.csv')
vhb_md = ProcessMetaData(vhb_md)

vhb_seq = SeqIO.parse(genomedir + "/vhb/sequences.fasta", "fasta")
vhb_sel = SelectGenomes(vhb_seq, vhb_md.index)

vhb_md = vhb_md.loc[vhb_sel[0]]
vhb_md['seq'] =  vhb_sel[1]

encvhb = [MakeSequenceEncoding(sq, 16*16*16) for sq in vhb_md['seq']]
encvhb = np.stack(encvhb)

dataSequence = DataSequenceLocal(encvhb,batchSize,Shuffle=False)

Model_vhb = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00vhb/model/encoder.h5')

Encoded_vhb = Model_vhb.predict(dataSequence)

vhb_md['Parameter-1'] = Encoded_vhb[2][:,0]
vhb_md['Parameter-2'] = Encoded_vhb[2][:,1]

vhb_md['virus'] = ['VHB' for _ in range(nSamples)]

##############################################################################
# Data loading 
###############################################################################

dnv_md = pd.read_csv(genomedir+'/dnv/sequences.csv')
dnv_md = ProcessMetaData(dnv_md)

dnv_seq = SeqIO.parse(genomedir + "/dnv/sequences.fasta", "fasta")
dnv_sel = SelectGenomes(dnv_seq, dnv_md.index)

dnv_md = dnv_md.loc[dnv_sel[0]]
dnv_md['seq'] =  dnv_sel[1]

encdnv = [MakeSequenceEncoding(sq, 24*24*24) for sq in dnv_md['seq']]
encdnv = np.stack(encdnv)

dataSequence = DataSequenceLocal(encdnv,batchSize,Shuffle=False)

Model_dnv = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00dnv/model/encoder.h5')

Encoded_dnv = Model_dnv.predict(dataSequence)

dnv_md['Parameter-1'] = Encoded_dnv[2][:,0]
dnv_md['Parameter-2'] = Encoded_dnv[2][:,1]

dnv_md['virus'] = ['DNV' for _ in range(nSamples)]


##############################################################################
# Data loading 
###############################################################################
outputPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/V3'
testIds = pd.read_csv(outputPath+'/samples/testmini_ids.csv')

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')
MetaData = MetaData.loc[testIds['ids']]

index = np.arange(MetaData.shape[0])
np.random.shuffle(index)

MetaData = MetaData.iloc[index[0:nSamples]]

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/singlenpy/'

ids = MetaData.index
dataSamps = np.array([basePath+'/'+val+'.npy' for val in ids])

dataSequence = DataSequenceFile(dataSamps,batchSize,Shuffle=False)

Model_sars = load_model('/media/tavo/storage/biologicalSequences/covidsr05/data/v4/00sars/model/encoder.h5')

Encoded_sars = Model_sars.predict(dataSequence)

MetaData['Parameter-1'] = Encoded_sars[2][:,0]
MetaData['Parameter-2'] = Encoded_sars[2][:,1]

MetaData['virus'] = ['SARS' for _ in range(nSamples)]

##############################################################################
# Data loading 
##############################################################################

columns = ['Parameter-1','Parameter-2','dayofyear']

virus = ['SARS','DNV','VHB','ROTA','INFM2','INFNEU','INFHA']

container = [MetaData[columns],dnv_md[columns],vhb_md[columns],
             rota_md[columns],infm2_md[columns],infneu_md[columns],
             infha_md[columns]]

maindict = {}

for val,sal in zip(virus,container):
    maindict[val] = sal.to_dict(orient='records')

with open('/media/tavo/storage/biologicalSequences/covidsr05/data/data.json', 'w') as f:
   json.dump(maindict, f)