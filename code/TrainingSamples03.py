#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:33:35 2024

@author: tavo
"""

import pandas as pd
from sklearn.model_selection import train_test_split

###############################################################################
# Loading packages 
###############################################################################

outputPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/V1'

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')

MetaData = MetaData[MetaData['effectivelength']>29645]
MetaData = MetaData[MetaData['effectivelength']<29760]

MetaData['partition'] = pd.qcut(MetaData['sunspots'],24,duplicates='drop')

###############################################################################
# Loading packages 
###############################################################################

X_train, X_test, _, _ = train_test_split(MetaData.index, 
                                         MetaData['sunspots'].values, 
                                         stratify=MetaData['sunspots'].values, 
                                         test_size=0.4, 
                                         random_state=42)

validationSamples = pd.DataFrame()
validationSamples['ids'] = X_test
validationSamples.to_csv(outputPath+'/folds/validationids.csv',index=False)

MetaDataV = MetaData.loc[X_test]

###############################################################################
# Loading packages 
###############################################################################

MetaData2 = MetaData.loc[X_train]

Xtrain, Xtest, _, _ = train_test_split(MetaData2.index, 
                                       MetaData2['sunspots'].values, 
                                       stratify=MetaData2['sunspots'].values, 
                                       test_size=0.1, 
                                       random_state=42)

trainSamples = pd.DataFrame()
trainSamples['ids'] = Xtrain
trainSamples.to_csv(outputPath+'/folds/trainids.csv',index=False)

testSamples = pd.DataFrame()
testSamples['ids'] = Xtest
testSamples.to_csv(outputPath+'/folds/testids.csv',index=False)

###############################################################################
# Loading packages 
###############################################################################

MiniTrain = MetaData.loc[Xtrain]

mtXtrain, mtXtest, _, _ = train_test_split(MiniTrain.index, 
                                           MiniTrain['sunspots'].values, 
                                           stratify=MiniTrain['sunspots'].values, 
                                           test_size=0.05, 
                                           random_state=42)

trainMiniSamples = pd.DataFrame()
trainMiniSamples['ids'] = mtXtest
trainMiniSamples.to_csv(outputPath+'/folds/trainmini_ids.csv',index=False)


MiniTest = MetaData.loc[Xtest]

mteXtrain, mteXtest, _, _ = train_test_split(MiniTest.index, 
                                             MiniTest['sunspots'].values, 
                                             stratify=MiniTest['sunspots'].values, 
                                             test_size=0.05, 
                                             random_state=42)

trainMiniSamples = pd.DataFrame()
trainMiniSamples['ids'] = mteXtest
trainMiniSamples.to_csv(outputPath+'/folds/testmini_ids.csv',index=False)

###############################################################################
# Loading packages 
###############################################################################


mitXtrain, mitXtest, _, _ = train_test_split(MiniTrain.index, 
                                           MiniTrain['sunspots'].values, 
                                           stratify=MiniTrain['sunspots'].values, 
                                           test_size=0.25, 
                                           random_state=42)

trainMiniSamples = pd.DataFrame()
trainMiniSamples['ids'] = mitXtest
trainMiniSamples.to_csv(outputPath+'/folds/trainmid_ids.csv',index=False)

miteXtrain, miteXtest, _, _ = train_test_split(MiniTest.index, 
                                             MiniTest['sunspots'].values, 
                                             stratify=MiniTest['sunspots'].values, 
                                             test_size=0.25, 
                                             random_state=42)

trainMiniSamples = pd.DataFrame()
trainMiniSamples['ids'] = miteXtest
trainMiniSamples.to_csv(outputPath+'/folds/testmid_ids.csv',index=False)

###############################################################################
# Loading packages 
###############################################################################

for k,val in enumerate(MetaData['partition'].unique()):
    
    localV = MetaDataV[MetaDataV['partition']==val]
    
    valByGroup = pd.DataFrame()
    valByGroup['ids'] = localV.index
    valByGroup.to_csv(outputPath+'/folds/groups/validationGroup0' +str(k)+'ids.csv',index=False)
    
    localT = MetaData2[MetaData2['partition']==val]
    
    Xtr, Xte, _, _ = train_test_split(localT.index, 
                                      localT['sunspots'].values, 
                                      stratify=localT['sunspots'].values, 
                                      test_size=0.15, 
                                      random_state=42)
    
    trainByGroup = pd.DataFrame()
    trainByGroup['ids'] = Xtr
    trainByGroup.to_csv(outputPath+'/folds/groups/trainGroup0' +str(k)+'ids.csv',index=False)
    
    testByGroup = pd.DataFrame()
    testByGroup['ids'] = Xte
    testByGroup.to_csv(outputPath+'/folds/groups/testGroup0' +str(k)+'ids.csv',index=False)
    