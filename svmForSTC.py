#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'svm for short text classification'

__author__ = 'lxp'

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
import random


#==========================随机选取n做训练集，实验证明10000比较好===========================
def sample(n = 10000):
	df = pd.read_table('vecdata.csv', sep = ',', encoding = 'utf-8', header = None)
	training = df.sample(n)
	return training


#========================sklearn做svm训练==============================================
#-----------------参数说明：_weight是正例的权重，负例权重设置为-1；m是训练样本数-----------
def svmtrain(_weight = 1, m = 10000):
	training = sample(n = m)
	x = training.drop(300, axis = 1)
	y = training[300]
	clf = svm.SVC( class_weight = {-1:1, 1: _weight})
	clf.fit(x, y)
	#把训练好的模型保存下来，名字为'train_model.m'
	joblib.dump(clf, "train_model.m")
	return

#=========================测试====================================
def accuracy():
	clf = joblib.load("train_model.m")
	#随机取20000条测试
	df = sample(20000)
	x = df.drop(300, axis = 1)
	y = df[300]
	m = n = TP = TN = 0
	for i in range(20000):
		if y.iloc[i] == 1:
			m = m + 1
			if clf.predict(x.iloc[i].values.reshape(1, -1)) == 1:
				TP = TP + 1
		else:
			n = n + 1
			if clf.predict(x.iloc[i].values.reshape(1, -1)) == -1:
				TN = TN + 1
	print('TP:', TP / m)
	print('TN:', TN / n)
	print('正确率：', (TP + TN) / 20000)
	return

#==========================================================================
#--------------------------------------------------------------------------
def test():
	#数据正负样本比例为1:3，给正例权重暂时设为3
	svmtrain(3)
	#-----测试10次------
	for i in range(10):
		accuracy()
	return

if __name__ == '__main__':
	test()