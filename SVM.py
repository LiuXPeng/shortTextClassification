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
def svmOneHotFindBestDivide(n = 1000):
	X, Y = fE.sample(n)
	y = fE.oneHotGet(Y)
	x = []
	for s in X:
		if s == 'e':
			x.append(-1)
		else:
			x.append(1)
	clf = svm.SVC()
	clf.fit(x, y)




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