#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'特征提取'

__author__ = 'lxp'


import featureExtract as fE
import numpy as np
from sklearn import svm
from sklearn.externals import joblib


#===============================================================================
#--------------------------------svm, one-hot-----------------------------------
def svmOneHot(lb = {'b':1, 't':1, 'e':1, 'm':-1}, n = 1000, W = 1, fe = 'one-hot', descend = None):
	X, Y = fE.sample(n)
	x = []
	for s in X:
		x.append(lb[s])
	if fe == 'one-hot':
		y = fE.oneHotGet(Y)
	if fe == 'tf-idf':
		y = fE.tfIdfGet(Y)
	if fe == 'word2vec':
		y = fE.word2vec(Y)
	if descend == 'pca':
		y = pcaGet(y)
	if descend == 'lda':
		y = ldaGet(y)
	clf = svm.SVC(class_weight = {-1:1, 1: W})
	clf.fit(x, y)
	joblib.dump(clf, "svmTrainModel.m")

	print('svm训练, label转换为:', lb, '\n训练集规模：', n, '\n权重：', W,'\n特征提取为:', fe, '\n降维方法：'， descend)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('svm训练, label转换为: ' + str(lb) + '\n训练集规模： ' + str(n) + '\n权重： ' + str(W) + '\n特征提取为: ' + str(fe) + '\n降维方法： ' + str(descend) + '\n')
	log.write('模型保存在： svmTrainModel.m中\n')
	log.close()

	return

def accuracy():
	




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


	return


if __name__ == '__main__':
	main()