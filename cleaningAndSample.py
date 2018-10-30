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



#================去掉数据中不需要的属性，将多分类变成二分类问题====================
def cleanData():
	f = open('uci-news-aggregator.csv', 'r', encoding='utf-8')
	g = open('label_text.txt', 'w', encoding = 'utf-8')
	g.truncate()
	g.close()
	g =open('label_text.txt', 'a', encoding = 'utf-8')
	log = open('log.txt', 'a', encoding = 'utf-8')
	line = f.readline()
	line = f.readline()

	b = 0
	t = 0
	e = 0
	m = 0
	count = 0
	while line:
		L = line.split(",")
		k = len(L)
		if k >= 8:
			text = L[k - 4] + ',' + L[1]
			for i in range(2, k - 6):
				text += ' ' + L[i]
			text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', text)
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

	print('b:', b, b / count)
	log.write('b:' + str(b) + str(b / count) + '\n')
	print('t:', t, t / count)
	log.write('t:' + str(t) + str(t / count) + '\n')
	print('e:', e, e / count)
	log.write('e:' + str(e) + str(e / count) + '\n')
	print('m:', m, m / count)
	log.write('m:' + str(m) + str(m / count) + '\n')
	print(count)
	log.write('count:' + str(count) + '\n')

	f.close()
	g.close()
	log.close()

	return


def main():
	print(datetime.date.today())
	f = open('log.txt', 'a', encoding = 'utf-8')
	f.write('\n=============================================\n')
	f.write(str(datetime.date.today()) + '\n')
	f.close()
	cleanData()
	return


if __name__ == '__main__':
	main()