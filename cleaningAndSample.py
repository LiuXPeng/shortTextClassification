#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'预处理kaggle新闻数据集：清洗，标签转换，抽取训练集、验证集'

__author__ = 'lxp'


from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import time
import datetime
import re
import random



#================去掉数据中不需要的属性，清洗、统计数据集====================
def cleanData():
	f = open('uci-news-aggregator.csv', 'r', encoding='utf-8')
	g = open('label_text.txt', 'w', encoding = 'utf-8')
	g.truncate()
	g.close()
	g =open('label_text.txt', 'a', encoding = 'utf-8')
	log = open('log.txt', 'a', encoding = 'utf-8')#日志文件
	line = f.readline()
	line = f.readline()

	#记录类别信息，统计
	b = 0
	t = 0
	e = 0
	m = 0
	count = 0
	while line:
		L = line.split(",")
		k = len(L)
		if k >= 8:#文本中有可能有分隔符，所以是大于等于，等于的话，丢弃八万条，太多了
			text = L[k - 4] + ',' + L[1]
			#--------------这个for循环，有时候因为publish那里也有分隔符，所有会把url等也写进去，所以需要正则化，按照url清洗一下
			for i in range(2, k - 6):
				text += ' ' + L[i]
			text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', text)#这里并没有清洗太干净，但不影响后面，因为主要是在分词做计算，所以符号、数字、空格不影响
			g.write(text + '\n')
			count += 1
			if L[k - 4] == 'b':
				b += 1
			if L[k - 4] == 't':
				t += 1
			if L[k - 4] == 'e':
				e += 1
			if L[k - 4] == 'm':
				m += 1

		line = f.readline()

	#---------------------------打印结果，写入日志-------------------------------------
	print('b:', b, b / count)
	log.write('b:' + str(b) + ', ' + str(b / count) + '\n')
	print('t:', t, t / count)
	log.write('t:' + str(t) + ', ' + str(t / count) + '\n')
	print('e:', e, e / count)
	log.write('e:' + str(e) + ', ' + str(e / count) + '\n')
	print('m:', m, m / count)
	log.write('m:' + str(m) + ', ' + str(m / count) + '\n')
	print(count)
	log.write('count:' + str(count) + '\n')

	f.close()
	g.close()
	log.close()

	return

#=================================抽取训练集，测试集，验证集========================
def splitData():
	f = open('label_text.txt', 'r', encoding = 'utf-8')
	line = f.readline()

	#--------------利用random，打乱list中的顺序，然后按照比例抽取----------
	dataSet = []
	while line:
		dataSet.append(line.split(','))
		line = f.readline()
	f.close()
	random.shuffle(dataSet)

	pos = int(len(dataSet) * 0.8)
	pos2 = int(len(dataSet) * 0.9)
	trainingSet = dataSet[:pos]
	testSet = dataSet[pos:pos2]
	validateSet = dataSet[pos2:]


	log = open('log.txt', 'a', encoding = 'utf-8')#日志文件

	#--------------------------训练集--------------------------
	#记录类别信息，统计
	b = 0
	t = 0
	e = 0
	m = 0
	count = 0
	g = open('trainingSet.txt', 'w', encoding = 'utf-8')
	g.truncate()
	g.close()
	g = open('trainingSet.txt', 'a', encoding = 'utf-8')
	for L in trainingSet:
		g.write(L[0] + ',' + L[1])
		count += 1
		if L[0] == 'b':
			b += 1
		if L[0] == 't':
			t += 1
		if L[0] == 'e':
			e += 1
		if L[0] == 'm':
			m += 1
	g.close()
	#--------------打印结果，写入日志-----------
	print('训练集：')
	log.write('训练集：' + '\n')
	print('b:', b, b / count)
	log.write('b:' + str(b) + ', ' + str(b / count) + '\n')
	print('t:', t, t / count)
	log.write('t:' + str(t) + ', ' + str(t / count) + '\n')
	print('e:', e, e / count)
	log.write('e:' + str(e) + ', ' + str(e / count) + '\n')
	print('m:', m, m / count)
	log.write('m:' + str(m) + ', ' + str(m / count) + '\n')
	print(count)
	log.write('count:' + str(count) + '\n')


	#---------------------------------测试集------------------------
	b = 0
	t = 0
	e = 0
	m = 0
	count = 0
	h = open('testSet.txt', 'w', encoding = 'utf-8')
	h.truncate()
	h.close()
	h = open('testSet.txt', 'a', encoding = 'utf-8')
	for L in testSet:
		h.write(L[0] + ',' + L[1])
		count += 1
		if L[0] == 'b':
			b += 1
		if L[0] == 't':
			t += 1
		if L[0] == 'e':
			e += 1
		if L[0] == 'm':
			m += 1
	h.close()
	print('测试集：')
	log.write('测试集：' + '\n')
	print('b:', b, b / count)
	log.write('b:' + str(b) + ', ' + str(b / count) + '\n')
	print('t:', t, t / count)
	log.write('t:' + str(t) + ', ' + str(t / count) + '\n')
	print('e:', e, e / count)
	log.write('e:' + str(e) + ', ' + str(e / count) + '\n')
	print('m:', m, m / count)
	log.write('m:' + str(m) + ', ' + str(m / count) + '\n')
	print(count)
	log.write('count:' + str(count) + '\n')


	#---------------------------------验证集------------------------
	b = 0
	t = 0
	e = 0
	m = 0
	count = 0
	h = open('validateSet.txt', 'w', encoding = 'utf-8')
	h.truncate()
	h.close()
	h = open('validateSet.txt', 'a', encoding = 'utf-8')
	for L in validateSet:
		h.write(L[0] + ',' + L[1])
		count += 1
		if L[0] == 'b':
			b += 1
		if L[0] == 't':
			t += 1
		if L[0] == 'e':
			e += 1
		if L[0] == 'm':
			m += 1
	h.close()
	print('验证集：')
	log.write('验证集：' + '\n')
	print('b:', b, b / count)
	log.write('b:' + str(b) + ', ' + str(b / count) + '\n')
	print('t:', t, t / count)
	log.write('t:' + str(t) + ', ' + str(t / count) + '\n')
	print('e:', e, e / count)
	log.write('e:' + str(e) + ', ' + str(e / count) + '\n')
	print('m:', m, m / count)
	log.write('m:' + str(m) + ', ' + str(m / count) + '\n')
	print(count)
	log.write('count:' + str(count) + '\n')
	log.close()

	return




def main():
	#--------------------------------每次运行，打印、写入时间戳-------------------------
	print(datetime.date.today())
	f = open('log.txt', 'a', encoding = 'utf-8')
	f.write('\n=============================================\n')
	f.write(str(datetime.date.today()) + '    cleaningAndSample.py' + '\n')
	f.close()

	#cleanData()
	#splitData()
	return


if __name__ == '__main__':
	main()