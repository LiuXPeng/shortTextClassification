#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'预处理kaggle新闻数据集'

__author__ = 'lxp'

from pandas import Series, DataFrame
import numpy as np
import pandas as pd

def cleanData():
	#去掉数据中不需要的属性，将多分类变成二分类问题
	x = pd.read_csv("uci-news-aggregator.csv")
	counts = x['CATEGORY'].value_counts()
	print("整个数据集类别统计", counts)

	del x['ID']
	del x['URL']
	del x['PUBLISHER']
	del x['STORY']
	del x['HOSTNAME']
	del x['TIMESTAMP']

	x['CATEGORY'].replace(['b', 't', 'e', 'm'], [-1, 1, -1, -1],inplace = True)
	#因为已经生成新的文件，因此这里将下面生成文件的代码注释
	#print(x)
	#x.to_csv('data.csv', sep=',', header=None, index=None)
	counts = x['CATEGORY'].value_counts()
	print("整个数据集二分类别统计", counts)
	return None

def sampleData(n = 100000):
	#抽取数据做训练集或测试集
	trainingData = x.sample(n)
	print("统计抽取样本",trainingData['CATEGORY'].value_counts())

	#这里将标签放在前面，文本放在后面
	trainingData.insert(0, 'LABEL', trainingData['CATEGORY'])
	del trainingData['CATEGORY']

	trainingData.to_csv('trainingData.txt', sep=',', header=None, index=None)
	return None

