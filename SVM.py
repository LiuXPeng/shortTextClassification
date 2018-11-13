#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'特征提取'

__author__ = 'lxp'


import featureExtract as fE
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import time
import datetime
import random


#===============================================================================
#-------------------------------------svm-----------------------------------
def svmOneHot(lb = {'b':1, 't':1, 'e':1, 'm':-1}, n = 1000, W = 1, fe = 'one-hot', descend = None):
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
		x = pcaGet(X)
	if descend == 'lda':
		x = ldaGet(X)
	clf = svm.SVC(class_weight = {-1:1, 1: W})
	clf.fit(x, y)
	joblib.dump(clf, "svmTrainModel.m")

	print('svm训练, label转换为:', lb, '\n训练集规模：', n, '\n权重：', W,'\n特征提取为:', fe, '\n降维方法：', descend)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('svm训练, label转换为: ' + str(lb) + '\n训练集规模： ' + str(n) + '\n权重： ' + str(W) + '\n特征提取为: ' + str(fe) + '\n降维方法： ' + str(descend) + '\n')
	log.write('模型保存在： svmTrainModel.m中\n---------------------\n')
	log.close()

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

	TP = 0
	FP = 0
	TN = 0
	FN = 0

	count = 0
	for data in dataSet:
		Y = data[0]
		X = [data[1].strip('\n')]
		
		count += 1
		if fe == 'one-hot':
			x = fE.oneHotGet(X)
		if fe == 'tf-idf':
			x = fE.tfIdfGet(X)
		if fe == 'word2vec':
			x = fE.word2vec(X)
		if descend == 'pca':
			x = pcaGet(X)
		if descend == 'lda':
			x = ldaGet(X)

		if clf.predict(x)[0] == lb[Y] and lb[Y] == 1:
			TP += 1
		if clf.predict(x)[0] == lb[Y] and lb[Y] == -1:
			TN += 1

		if clf.predict(x)[0] != lb[Y] and lb[Y] == 1:
			FN += 1
		if clf.predict(x)[0] != lb[Y] and lb[Y] == -1:
			FP += 1

		if count % 500 == 0:
			print(count)

		if count % 1000 == 0:
			break

	#count = TP + TN + FN + FP
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	F1 = 2 * TP / (count + TP - TN)
	print('svm测试\nlabel转换为:', lb, '\n特征提取为:', fe, '\n降维方法：', descend)
	print('TP:', TP, 'TN:', TN, 'FN:', FN, 'FP:', FP)
	print('总计：', count)
	print('精度:', TP + TN, ', ', (TP + TN) / count)
	print('查准率：', precision)
	print('查全率：', recall)
	print('F1：', F1)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('svm测试\nlabel转换为: ' + str(lb)  + '\n特征提取为: ' + str(fe) + '\n降维方法： ' + str(descend) + '\n')
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
	f.write(str(datetime.date.today()) +'    SVM.py' + '\n')
	f.close()

	#--------------------------------------------------------
	for i in range(10):
		k = 1 + i * 0.5
		svmOneHot(lb = {'b':1, 't':-1, 'e':1, 'm':1}, n = 1000, W = k, fe = 'one-hot', descend = None)
		print('---------------------------------------')
		accuracy(lb = {'b':1, 't':-1, 'e':1, 'm':1}, fe = 'one-hot', descend = None)
		print('######################################')

	for i in range(10):
		k = 1 + i * 0.5
		svmOneHot(lb = {'b':1, 't':1, 'e':-1, 'm':1}, n = 1000, W = k, fe = 'one-hot', descend = None)
		print('---------------------------------------')
		accuracy(lb = {'b':1, 't':1, 'e':-1, 'm':1}, fe = 'one-hot', descend = None)
		print('######################################')



	return


if __name__ == '__main__':
	main()