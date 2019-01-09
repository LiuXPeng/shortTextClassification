#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'svm'

__author__ = 'lxp'


import featureExtract as fE
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import time
import datetime
import random
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


#===============================================================================
#-------------------------------------svm-----------------------------------
def svmTrain(lb = {'b':1, 't':1, 'e':1, 'm':-1}, n = 1000, W = 1, fe = 'one-hot', descend = None):
	Y, X = fE.sample(n)
	y = []
	for s in Y:
		y.append(lb[s])
	if fe == 'one-hot':
		x = fE.oneHotGet(X)
	if fe == 'tf-idf':
		x = fE.tfIdfGet(X)
	if fe == 'word2vec':
		x = fE.word2vec(X)

	if descend == 'pca':
		x = fE.pcaGet(x)
	if descend == 'lda':
		x = fE.ldaGet(x)

	clf = LogisticRegression()
	clf.fit(x, y)
	joblib.dump(clf, "svmTrainModel.m")

	print('svm训练, label转换为:', lb, '\n训练集规模：', n, '\n权重：', W,'\n特征提取为:', fe, '\n降维方法：', descend)
	
	return

def accuracy(lb = {'b':1, 't':1, 'e':1, 'm':-1}, fe = 'one-hot', descend = None):
	clf = joblib.load("svmTrainModel.m")
	f = open('label_segmentation_test.txt', 'r', encoding = 'utf-8')
	line = f.readline()
	dataSet = []
	while line:
		dataSet.append(line.split(','))
		line = f.readline()
	f.close()

	random.shuffle(dataSet)

	count = 3000

	X = []
	Y = []

	for i in range(count):
		Y.append(dataSet[i][0])
		X.append(dataSet[i][1].strip('\n'))

	y = []
	for s in Y:
		y.append(lb[s])
	if fe == 'one-hot':
		x = fE.oneHotGet(X)
	if fe == 'tf-idf':
		x = fE.tfIdfGet(X)
	if fe == 'word2vec':
		x = fE.word2vec(X)
	if descend == 'pca':
		x = fE.pcaGet(x)
	if descend == 'lda':
		x = fE.ldaGet(x)


	n = 0

	for i in range(count):

		if fe != 'word2vec':
			x[i] = np.array(x[i])
		res = clf.predict(x[i].reshape(1, -1))

		if res == y[i]:
			n += 1


		if i % 50 == 0:
			print(res)


	#count = TP + TN + FN + FP
	print('svm测试\nlabel转换为:', lb, '\n特征提取为:', fe, '\n降维方法：', descend)
	print('总计：', count)
	print('精度:', n / count)



	return



#===============================================================================
#===============================================================================
def main():
	#--------------------------------每次运行，打印、写入时间戳-------------------------
	print(datetime.date.today())

	#--------------------------------------------------------
	temp = 'word2vec'
	Des = None
	
	svmTrain(lb = {'b':0, 't':1, 'e':2, 'm':3}, n = 10000, fe = temp, descend = Des) 
	print('---------------------------------------')
	accuracy(lb = {'b':0, 't':1, 'e':2, 'm':3}, fe = temp, descend = Des)
	print('######################################')


	return


if __name__ == '__main__':
	main()