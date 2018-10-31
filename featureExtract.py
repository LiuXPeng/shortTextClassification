#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'特征提取'

__author__ = 'lxp'


import nltk
from nltk.tokenize import word_tokenize



#============================================================


#==========================one-hot编码==============================
def oneHotDict():
	f = open('trainingSet', 'r', encoding = 'utf-8')
	line = f.readline()
	count = 0
	dictionary = {}
	while line:
		L = line.split(',')
		line = f.readline()
		pass
	f.close()



#===============================================================================
#===============================================================================
def main():
	#--------------------------------每次运行，打印、写入时间戳-------------------------
	print(datetime.date.today())
	f = open('log.txt', 'a', encoding = 'utf-8')
	f.write('\n=============================================\n')
	f.write(str(datetime.date.today()) + '\n')
	f.close()

	#--------------------------------------------------------

	return


if __name__ == '__main__':
	main()