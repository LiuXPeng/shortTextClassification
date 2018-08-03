#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'预处理kaggle新闻数据集'

__author__ = 'lxp'

from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import csv
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


#================去掉数据中不需要的属性，将多分类变成二分类问题====================
def cleanData():
	x = pd.read_csv("uci-news-aggregator.csv")
	counts = x['CATEGORY'].value_counts()
	print("整个数据集类别统计", counts)

	#删除无关属性
	del x['ID']
	del x['URL']
	del x['PUBLISHER']
	del x['STORY']
	del x['HOSTNAME']
	del x['TIMESTAMP']
	x.dropna(axis = 0)

	x['CATEGORY'].replace(['b', 't', 'e', 'm'], [-1, 1, -1, -1],inplace = True)
	category = x['CATEGORY']
	x.drop(labels=['CATEGORY'], axis=1,inplace = True)
	x.insert(0, 'CATEGORY', category)

	#因为已经生成新的文件，因此可以这里将下面生成文件的代码注释
	x.to_csv('notCleandata.csv', sep=',', header=None, index=None, encoding='utf-8')
	counts = x['CATEGORY'].value_counts()
	print("整个数据集二分类别统计", counts)

	#-------------在处理时发现，这个数据集有几百数据分隔符和其余不一样，pandas读取得数据有部分读不了
	#但是，写入时候会原封不动再写进去，因此，通过每一行前几个字符，踢去这些数据-----------------------------
	f = open('notCleandata.csv', 'r', encoding='utf-8')
	g = open('data.txt', 'w+')
	g.truncate()  # 清空文件，防止测试时候不停加入
	g.close()
	g = open('data.txt', 'a', encoding='utf-8')
	line = f.readline()
	while line:
		if (line[0] == '1' and line[1] == ',') or (line[0] == '-' and line[1] == '1' and line[2] == ','):
			g.write(line)
		line = f.readline()
	f.close()
	g.close()
	return 0

#======================部分算法需要分词，这里就对数据进行分词，格式为：标签，分词（分词以空格隔开）=======================
def stemming():
	#f = open('data.csv', encoding= 'gb18030', errors= 'ignore')
	f = open('data.txt', 'r', encoding='utf-8')
	g = open('segmentation.txt', 'w+')
	g.truncate()#清空文件，防止测试时候不停加入
	g.close()
	g = open('segmentation.txt', 'a', encoding='utf-8')
	line = f.readline()
	#停词库
	notWords = ['i', 'me', 'my', 'myself', 'we', 'our',\
	 'ours', 'ourselves', 'you', 'your', 'yours',\
	  'yourself', 'yourselves', 'he', 'him', 'his',\
	   'himself', 'she', 'her', 'hers', 'herself',\
	    'it', 'its', 'itself', 'they', 'them', 'their',\
	     'theirs', 'themselves', 'what', 'which',\
	      'who', 'whom', 'this', 'that', 'these',\
	       'those', 'am','is', 'are', 'was', 'were',\
	        'be', 'been', 'being','have', 'has', 'had',\
	         'having', 'do', 'does', 'did', 'doing',\
	          'a', 'an', 'the', 'and', 'but', 'if', 'or',\
	           'because', 'as', 'until', 'while', 'of',\
	            'at', 'by', 'for', 'with', 'about', 'against',\
	             'between', 'into', 'through', 'during',\
	              'before', 'after', 'above', 'below', 'to',\
	               'from', 'up', 'down', 'in', 'out', 'on',\
	                'off', 'over', 'under', 'again', 'further',\
	                 'then', 'once', 'here', 'there', 'when',\
	                  'where', 'why', 'how', 'all', 'any', 'both',\
	                   'each', 'few', 'more', 'most', 'other',\
	                    'some', 'such', 'no', 'nor', 'not', 'only',\
	                     'own', 'same', 'so', 'than', 'too', 'very',\
	                      's', 't', 'can', 'will', 'just', 'don',\
	                       'should', 'now', ',', '.', ':', ';', '?',\
	                        '(', ')', '[', ']', '&', '!', '*', '@',\
	                         '#', '$', '%']
	while line:
		L = line.split(',', 1)
		line = f.readline()
		g.write(L[0])
		g.write(',')
		#去停词
		words = word_tokenize(L[1].lower())
		for word in words:
			if not word.isalpha:
				continue
			if word in notWords:
				continue
			g.write(word)
			g.write(' ')
		g.write('\n')
	f.close()
	g.close()
	return

#====================把所有标签删除，分词写到一个文件，并按照标签顺序写入，为word2vec训练准备语料=================
#因为后面的程序是把label和data分开处理的，所以这里也分开写
def makeWordBag():
	f = open('segmentation.txt', 'r', encoding='utf-8')
	g = open('sentences.txt', 'w+')
	g.truncate()#清空文件，防止测试时候不停加入
	g.close()
	g = open('sentences.txt', 'a', encoding='utf-8')
	h = open('labels.txt', 'w+')
	h.truncate()#清空文件，防止测试时候不停加入
	h.close()
	h = open('labels.txt', 'a', encoding='utf-8')
	line = f.readline()
	#--------------先写入+标签的数据---------
	while line:
		L = line.split(',', 1)
		line = f.readline()
		if L[0][0] == '1':
			g.write(L[1])
			h.write('1' + '\n')
	f.close()

	#--------------再写入-标签的数据----------------
	f = open('segmentation.txt', 'r', encoding='utf-8')
	line = f.readline()
	while line:
		L = line.split(',', 1)
		line = f.readline()
		if L[0][0] == '-':
			g.write(L[1])
			h.write('-1' + '\n')
			
	f.close()	
	g.close()
	h.close()
	return

#================word2vec训练过程=========================
def w2v():
	sentences = LineSentence('sentences.txt')
	model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=4)
	model.save('42w.model')#保存模型
	print(model.wv['computer'])
	return

#=========================把整个文件中所有数据做sent2vec==========================
def sent2vec():
	f = open('sentences.txt', 'r', encoding='utf-8')
	line = f.readline()
	g = open('vecs.txt', 'w+')
	g.truncate()#清空文件，防止测试时候不停加入
	g.close()
	g = open('vecs.txt', 'a', encoding='utf-8')
	#这里用的是谷歌新闻训练好的一个语料，这个地方比较吃内存
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
	#----------------------每个句子中的分词做word2vec,再去平均-------------------------
	while line:
		L = line.split(' ')
		line = f.readline()
		m = np.zeros(300)
		n = 0
		for l in L:
			try:
				k = model.wv[l]
				m = m + k
				n = n + 1
			except:
				continue
		if n != 0:
			#取平均
			m = m / n
		for i in m:
			g.write(str(i))
			g.write(' ')
		g.write('\n')
	f.close()
	g.close()

	#--------------------把vec和labels合为一个文件vecdata.csv， vecs在前，label在后-----------------
	df = pd.read_table('vecs.txt', sep = ' ', encoding = 'utf-8', header = None)
	del df[300]
	df1 = pd.read_table('labels.txt', sep = ' ', encoding = 'utf-8', header = None)
	df['label'] = df1
	df.to_csv('vecdata.csv', index = False,header = False, sep = ',')

	return


#===================================================
#---------------------------------------------------
def test():
	cleanData()
	stemming()
	makeWordBag()
	w2v()
	#实验发现42万条的word2vec训练太少了，因此采用了谷歌新闻语料
	sent2vec()
	return

if __name__ == '__main__':
	test()