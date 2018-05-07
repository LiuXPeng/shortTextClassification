#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'预处理kaggle新闻数据集'

__author__ = 'lxp'

from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import csv

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
	x.dropna(axis = 0)

	x['CATEGORY'].replace(['b', 't', 'e', 'm'], [-1, 1, -1, -1],inplace = True)
	category = x['CATEGORY']
	x.drop(labels=['CATEGORY'], axis=1,inplace = True)
	x.insert(0, 'CATEGORY', category)
	#因为已经生成新的文件，因此这里将下面生成文件的代码注释
	#print(x)
	x.to_csv('notCleandata.csv', sep=',', header=None, index=None, encoding='utf-8')
	counts = x['CATEGORY'].value_counts()
	print("整个数据集二分类别统计", counts)
	f = open('notCleandata.csv', 'r', encoding='utf-8')
	g = open('data.txt', 'w+')
	g.truncate()  # 清空文件，防止测试时候不停加入
	g.close()
	g = open('data.txt', 'a', encoding='utf-8')
	line = f.readline()
	while line:
		if (line[0] == '1' and line[1] == ',') or (line[0] == '-' and line[1] == '1' and line[2] == ','):
			g.write(line)
		line = f.readline()
	f.close()
	g.close()
	return 0

def sampleData(n = 100000):
	#抽取数据做训练集
	sampleData = x.sample(n)
	print("统计抽取样本",trainingData['CATEGORY'].value_counts())

	#这里将标签放在前面，文本放在后面
	trainingData.insert(0, 'LABEL', trainingData['CATEGORY'])
	del trainingData['CATEGORY']

	trainingData.to_csv('trainingData.txt', sep=',', header=None, index=None)
	return sampleData

def stemming():
	#f = open('data.csv', encoding= 'gb18030', errors= 'ignore')
	f = open('data.txt', 'r', encoding='utf-8')
	g = open('segmentation.txt', 'w+')
	g.truncate()#清空文件，防止测试时候不停加入
	g.close()
	g = open('segmentation.txt', 'a', encoding='utf-8')
	line = f.readline()
	notWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
	while line:
		L = line.split(',', 1)
		line = f.readline()
		g.write(L[0])
		g.write(',')
		words = word_tokenize(L[1].lower())
		for word in words:
			if not word.isalpha:
				continue
			if word in notWords:
				continue
			g.write(word)
			g.write(' ')
		g.write('\n')
	f.close()
	g.close()
	return 0

stemming()
#cleanData()
