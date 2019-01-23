#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'logic regression'

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
#-------------------------------------lr-----------------------------------
def lrTrain(lb = {'b':1, 't':1, 'e':1, 'm':-1}, n = 1000,  fe = 'one-hot', descend = None):
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

	clf = LogisticRegression(C = 1, penalty='l1', tol=0.01, class_weight = 'balanced', solver='saga')
	clf.fit(x, y)
	joblib.dump(clf, "lrTrainModel.m")

	print('lr训练, label转换为:', lb, '\n训练集规模：', n, '\n特征提取为:', fe, '\n降维方法：', descend)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('lr训练, label转换为: ' + str(lb) + '\n训练集规模： ' + str(n) + '\n特征提取为: ' + str(fe) + '\n降维方法： ' + str(descend) + '\n')
	log.write('模型保存在： lrTrainModel.m中\n---------------------\n')
	log.close()

	return

def accuracy(lb = {'b':1, 't':1, 'e':1, 'm':-1}, fe = 'one-hot', descend = None):
	clf = joblib.load("lrTrainModel.m")
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

	TP = 1
	FP = 1
	TN = 1
	FN = 1


	for i in range(count):

		if fe != 'word2vec':
			x[i] = np.array(x[i])
		res = clf.predict(x[i].reshape(1, -1))

		if res == y[i] and y[i] == 1:
			TP += 1
		if res == y[i] and y[i] == -1:
			TN += 1
		if res != y[i] and y[i] == 1:
			FN += 1
		if res != y[i] and y[i] == -1:
			FP += 1

		if i % 500 == 0:
			print(i)


	#count = TP + TN + FN + FP
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	F1 = 2 * TP / (count + TP - TN)
	print('lr测试\nlabel转换为:', lb, '\n特征提取为:', fe, '\n降维方法：', descend)
	print('TP:', TP, 'TN:', TN, 'FN:', FN, 'FP:', FP)
	print('总计：', count)
	print('精度:', TP + TN, ', ', (TP + TN) / count)
	print('查准率：', precision)
	print('查全率：', recall)
	print('F1：', F1)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('lr测试\nlabel转换为: ' + str(lb)  + '\n特征提取为: ' + str(fe) + '\n降维方法： ' + str(descend) + '\n')
	log.write('TP: ' + str(TP) + ', TN: ' + str(TN) + ', FN:' + str(FN) + ', FP:' + str(FP) + '\n')
	log.write('总计： ' + str(count) + '\n')
	log.write('精度: ' + str(TP + TN) + ', ' + str((TP + TN) / count) + '\n')
	log.write('查准率： ' + str(precision) + '\n')
	log.write('查全率： ' + str(recall) + '\n')
	log.write('F1： ' + str(F1) + '\n-------------------\n')
	log.close()



	return



#===============================================================================
#===============================================================================
def main():
	#--------------------------------每次运行，打印、写入时间戳-------------------------
	print(datetime.date.today())
	f = open('log.txt', 'a', encoding = 'utf-8')
	f.write('\n=============================================\n')
	f.write(str(datetime.date.today()) +'    LR.py' + '\n')
	f.close()

	#--------------------------------------------------------
	temp = 'word2vec'
	Des = None
	
	#lrTrain(lb = {'b':-1, 't':1, 'e':1, 'm':1}, n = 10000, fe = temp, descend = Des) 
	#print('---------------------------------------')
	#accuracy(lb = {'b':-1, 't':1, 'e':1, 'm':1}, fe = temp, descend = Des)
	#print('######################################')


	lrTrain(lb = {'b':-1, 't':-1, 'e':1, 'm':1}, n = 10000, fe = temp, descend = Des)
	print('---------------------------------------')
	accuracy(lb = {'b':-1, 't':-1, 'e':1, 'm':1}, fe = temp, descend = Des)
	print('######################################')


	lrTrain(lb = {'b':-1, 't':1, 'e':-1, 'm':1}, n = 10000, fe = temp, descend = Des)
	print('---------------------------------------')
	accuracy(lb = {'b':-1, 't':1, 'e':-1, 'm':1}, fe = temp, descend = Des)
	print('######################################')


	lrTrain(lb = {'b':-1, 't':1, 'e':1, 'm':-1}, n = 10000,  fe = temp, descend = Des)
	print('---------------------------------------')
	accuracy(lb = {'b':-1, 't':1, 'e':1, 'm':-1}, fe = temp, descend = Des)
	print('######################################')

	#--------------------------------------------------------
	temp = 'tf-idf'
	
	#lrTrain(lb = {'b':-1, 't':1, 'e':1, 'm':1}, n = 10000, fe = temp, descend = Des) 
	#print('---------------------------------------')
	#accuracy(lb = {'b':-1, 't':1, 'e':1, 'm':1}, fe = temp, descend = Des)
	#print('######################################')


	lrTrain(lb = {'b':-1, 't':-1, 'e':1, 'm':1}, n = 10000, fe = temp, descend = Des)
	print('---------------------------------------')
	accuracy(lb = {'b':-1, 't':-1, 'e':1, 'm':1}, fe = temp, descend = Des)
	print('######################################')


	lrTrain(lb = {'b':-1, 't':1, 'e':-1, 'm':1}, n = 10000, fe = temp, descend = Des)
	print('---------------------------------------')
	accuracy(lb = {'b':-1, 't':1, 'e':-1, 'm':1}, fe = temp, descend = Des)
	print('######################################')


	lrTrain(lb = {'b':-1, 't':1, 'e':1, 'm':-1}, n = 10000,  fe = temp, descend = Des)
	print('---------------------------------------')
	accuracy(lb = {'b':-1, 't':1, 'e':1, 'm':-1}, fe = temp, descend = Des)
	print('######################################')


	return


if __name__ == '__main__':
	main()
