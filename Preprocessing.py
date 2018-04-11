#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'预处理kaggle新闻数据集'

__author__ = 'lxp'

from pandas import Series, DataFrame
import numpy as np
import pandas as pd

x = pd.read_csv("uci-news-aggregator.csv")
counts = x['CATEGORY'].value_counts()
print(counts)

del x['ID']
del x['URL']
del x['PUBLISHER']
del x['STORY']
del x['HOSTNAME']
del x['TIMESTAMP']
#x.to_csv('a.csv', sep=',', header=None, index=None)
x['CATEGORY'].replace(['b', 't', 'e', 'm'], [-1, 1, -1, -1],inplace = True)
#print(x)
#x.to_csv('data.csv', sep=',', header=None, index=None)
counts = x['CATEGORY'].value_counts()
print(counts)
trainingData = x.sample(n = 100000)
print(trainingData['CATEGORY'].value_counts())
trainingData.insert(0, 'LABEL', trainingData['CATEGORY'])
del trainingData['CATEGORY']
print(trainingData)
trainingData.to_csv('trainingData.txt', sep=',', header=None, index=None)
