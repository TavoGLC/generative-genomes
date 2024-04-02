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

outputPath = '/media/tavo/storage/biologicalSequences/covidsr05/data/V2'

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
                                         test_size=0.1, 
                                         random_state=42)

trainSamples = pd.DataFrame()
trainSamples['ids'] = X_train
trainSamples.to_csv(outputPath+'/trainids.csv',index=False)

testSamples = pd.DataFrame()
testSamples['ids'] = X_test
testSamples.to_csv(outputPath+'/testids.csv',index=False)

