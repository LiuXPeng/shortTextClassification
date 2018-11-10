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
	log.write('模型保存在： svmTrainModel.m中\n')
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

	count = 0
	right = 0

	for data in dataSet:
		count += 1
		Y = data[0]
		X = [data[1].strip('\n')]
		
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

		if clf.predict(x)[0] == lb[Y]:
			right += 1

		if count % 5000 == 0:
			break

	print('svm测试\nlabel转换为:', lb, '\n特征提取为:', fe, '\n降维方法：', descend)
	print('总计：', count, '分类正确', right, right / count)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('svm测试\nlabel转换为: ' + str(lb)  + '\n特征提取为: ' + str(fe) + '\n降维方法： ' + str(descend) + '\n')
	log.write('总计：' + str(count) + ', 分类正确' + str(right) + ', '  + str(right / count))
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
	#svmOneHot()
	accuracy()



	return


if __name__ == '__main__':
	main()