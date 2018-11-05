#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'特征提取'

__author__ = 'lxp'


import time
import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
import pickle
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
import random
import math
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence



#============================================================
#================================================数据清洗================================================

#停词表构建
def load_stopwords():
	stop_words = nltk.corpus.stopwords.words('english')
	stop_words.extend(['this','that','the','might','have','been','from',
                'but','they','will','has','having','had','how','went'
                'were','why','and','still','his','her','was','its','per','cent',
                'a','able','about','across','after','all','almost','also','am','among',
                'an','and','any','are','as','at','be','because','been','but','by','can',
                'cannot','could','dear','did','do','does','either','else','ever','every',
                'for','from','get','got','had','has','have','he','her','hers','him','his',
                'how','however','i','if','in','into','is','it','its','just','least','let',
                'like','likely','may','me','might','most','must','my','neither','nor',
                'not','of','off','often','on','only','or','other','our','own','rather','said',
                'say','says','she','should','since','so','some','than','that','the','their',
                'them','then','there','these','they','this','tis','to','too','twas','us',
                'wants','was','we','were','what','when','where','which','while','who',
                'whom','why','will','with','would','yet','you','your','ve','re','rt', 'retweet', '#fuckem', '#fuck',
                'fuck', 'ya', 'yall', 'yay', 'youre', 'youve', 'ass','factbox', 'com', '&lt', 'th',
                'retweeting', 'dick', 'fuckin', 'shit', 'via', 'fucking', 'shocker', 'wtf', 'hey', 'ooh', 'rt&amp', '&amp',
                '#retweet', 'retweet', 'goooooooooo', 'hellooo', 'gooo', 'fucks', 'fucka', 'bitch', 'wey', 'sooo', 'helloooooo', 'lol', 'smfh'])
	stop_words = set(stop_words)
	return stop_words

#===============================正则化清洗推文================================
def normalize_text(text):
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
	text = re.sub('@[^\s]+','', text)
	text = re.sub('#([^\s]+)', '', text)
	text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', text)
	text = re.sub('[\d]','', text)
	text = text.replace(".", '')
	text = text.replace("'", ' ')
	text = text.replace("\"", ' ')
	#text = text.replace("-", " ")
	#normalize some utf8 encoding
	text = text.replace("\x9d",' ').replace("\x8c",' ')
	text = text.replace("\xa0",' ')
	text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace("\xf0\x9f\x91\x8d\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f",' ').replace("\x91\x8d",' ')
	text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",' ').replace("\xf0",' ').replace('\xf0x9f','').replace("\x9f\x91\x8d",' ').replace("\x87\xba\x87\xb8",' ')	
	text = text.replace("\xe2\x80\x94",' ').replace("\x9d\xa4",' ').replace("\x96\x91",' ').replace("\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8",' ')
	text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace("\xe2\x80\x99", ' ').replace("\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
	text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace("\xc2\xab", " ").replace("\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")
	text = re.sub("[^a-zA-Z]", " ", text)
	return text


#===============================分词、去停词==============================
def nltk_tokenize(text):
	tokens = []
	features = []
	tokens = text.split()
	stop_words = load_stopwords()
	for word in tokens:
			if word.lower() not in stop_words and len(word) > 2:
				features.append(word)
	return features


#标签、分词文件
def label_segmentation():
	f = open('trainingSet.txt', 'r', encoding = 'utf-8')
	line = f.readline()
	g = open('label_segmentation.txt', 'w', encoding = 'utf-8')
	g.truncate()
	g.close()
	g = open('label_segmentation.txt', 'a', encoding = 'utf-8')

	while line:
		L = line.split(',')
		line = f.readline()
		g.write(L[0] + ',')
		text = normalize_text(L[1])
		words = nltk_tokenize(text)
		g.write(' '.join(words) + '\n')

	g.close()
	f.close()

	print('标签，分词文件')
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('完成标签、分词文件： label_segmentation.txt' + '\n')
	log.close()

	return

#==================================================================
#==========================one-hot编码==============================

#-----------------------------创建字典---------------------
def oneHotDict():
	f = open('trainingSet.txt', 'r', encoding = 'utf-8')
	line = f.readline()
	count = 0
	dictionary = {}
	while line:
		L = line.split(',')
		line = f.readline()
		text = normalize_text(L[1])
		words = nltk_tokenize(text)
		for word in words:
			if word not in dictionary:
				dictionary[word] = count#编码时做索引
				count += 1

	f.close()

	#把结果存到文件中
	g = open('dictionary.plk', 'wb')
	pickle.dump(dictionary, g)
	g.close()

	print(count)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('one-hot编码维度' + str(count) + '\n')
	log.close()

	return

#--------------------------获取n条数据的one-hot------------------
def oneHotGet(Y):
	n = len(Y)
	f = open('dictionary.plk', 'rb')
	dictionary = pickle.load(f)
	f.close()

	Z = []
	for text in Y:
		temp = [0] * 67216
		words = text.split(' ')
		for word in words:
			if word in dictionary:
				temp[dictionary[word]] += 1
		Z.append(temp)

	print('抽取' + str(n) + '条one-hot编码')
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('抽取' + str(n) + '条one-hot编码\n')
	log.close()

	return Z


#测试one-hot编码中的全零向量
def oneHotStatistics():
	f = open('testSet.txt', 'r', encoding = 'utf-8')
	line = f.readline()
	
	#这里调用oneHotGet()函数，文件不停打开加载关闭太慢，所以直接写
	g = open('dictionary.plk', 'rb')
	dictionary = pickle.load(g)
	g.close()

	count = 0
	m = 0
	while line:
		count += 1
		L = line.split(',')
		line = f.readline()
		text = normalize_text(L[1])
		words = nltk_tokenize(text)
		for word in words:
			if word in dictionary:
				m += 1
				break

	f.close()

	print('零向量共计： ', count - m, (count - m) / count)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('one-hot零向量总计： ' + str(count - m) + ', ' + str((count - m) / count) + '\n')
	log.close()

	return


#=========================================================================
#==================================tf-idf===============================

#----------------------------建立tf-idf字典---------------------------
def tfIdfDict():
	f = open('label_segmentation.txt', 'r', encoding = 'utf-8')
	line = f.readline()
	dictionary = {}
	count = 0
	while line:
		line = line.strip('\n')#特别注意这个换行符问题！
		L = line.split(',')
		words = L[1].split(' ')
		line = f.readline()
		for word in words:
			if word in dictionary:
				dictionary[word] += 1
			else:
				dictionary[word] = 1
				count += 1

	print(count)

	f.close()

	g = open('idfDDictionary.plk', 'wb')
	pickle.dump(dictionary, g)
	g.close()

	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('计算idf中，每个词在整个文本中出现的次数，也就是包含词的文章数，即idf分母，按照dict存在idfDDictionary.plk' + '\n')
	log.close()

	return

#--------------------------获取n条数据的tf-idf------------------
def tfIdfGet(Y):
	n = len(Y)
	f = open('dictionary.plk', 'rb')
	dictionary = pickle.load(f)
	f.close()
	g= open('idfDDictionary.plk', 'rb')
	idfDictionary = pickle.load(g)
	g.close()

	Z = []
	for text in Y:
		temp = [0] * 67216
		words = text.split(' ')
		for word in words:
			if word in dictionary:
				tf = 1 / len(words)
				idf = math.log(337930 / idfDictionary[word])
				temp[dictionary[word]] += (tf * idf)
		Z.append(temp)

	print('抽取' + str(n) + '条tf-idf编码\n')
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('抽取' + str(n) + '条tf-idf编码\n')
	log.close()

	return  Z



#===================================PCA=========================================
#----------------------------------pca训练----------------------------------
def pcaTrain(X, n = 600):
	m = len(X)
	pca = PCA(n_components = n)

	pca.fit(X)

	g = open('pca.plk', 'wb')
	pickle.dump(pca, g)
	g.close()

	print("pca降维，维度：", n, '训练集规模：', m)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('pca降维， 维度：' + str(n) + ', 训练集规模： ' + str(m) + '\n')
	log.write('模型保存在： pca.plk中\n')
	log.close()

	return

def pcaGet(X):
	g= open('pca.plk', 'rb')
	pca = pickle.load(g)
	g.close()

	newX = pca.fit_transform(X)
	print('pca降维')
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('pca降维\n')
	log.close()
	return newX


#===================================LDA=========================================
#---------------------------------lda训练---------------------------------
def ldaTrain(X, n = 4):
	m = len(X)
	lda = LatentDirichletAllocation(n_components = n)

	lda.fit(X)

	g = open('lda.plk', 'wb')
	pickle.dump(lda, g)
	g.close()

	print("lda降维，维度：", n, '训练集规模：', m)
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('lda降维， 维度：' + str(n) + ', 训练集规模： ' + str(m) + '\n')
	log.write('模型保存在： lda.plk中\n')
	log.close()

	return


def ldaGet(X):
	g= open('lda.plk', 'rb')
	lda = pickle.load(g)
	g.close()

	newX = lda.fit_transform(X)

	print('lda降维')
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('lda降维\n')
	log.close()
	return newX

#============================================================================
#-------------------------------word2vec----------------------------------
def word2vec(X):
	newX = []
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
	for text in X:
		words = text.strip('\n').split(' ')
		m = np.zeros(300)
		n = 0
		for word in words:
			try:
				k = model.wv[word]
				m = m + k
				n = n + 1
			except:
				continue
		if n != 0:
			#取平均
			m = m / n
		newX.append(m)

	print('word2vec')
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('word2vec\n')
	log.close()


	return newX


#==========================随机抽取数据，label和数据以list形式返回=========================
def sample(n = 5):
	f = open('label_segmentation.txt', 'r', encoding = 'utf-8')
	line = f.readline()

	#--------------利用random，打乱list中的顺序，然后按照比例抽取----------
	dataSet = []
	while line:
		dataSet.append(line.split(','))
		line = f.readline()
	f.close()
	random.shuffle(dataSet)

	X = []
	Y = []

	for i in range(n):
		X.append(dataSet[i][0])
		Y.append(dataSet[i][1].strip('\n'))

	print('随机抽取' + str(n) + '条数据')
	log = open('log.txt', 'a', encoding = 'utf-8')
	log.write('随机抽取：' + str(n) + '条数据\n')
	log.close()

	return X,Y




#===============================================================================
#===============================================================================
def main():
	#--------------------------------每次运行，打印、写入时间戳-------------------------
	print(datetime.date.today())
	f = open('log.txt', 'a', encoding = 'utf-8')
	f.write('\n=============================================\n')
	f.write(str(datetime.date.today()) +'    featureExtract.py' + '\n')
	f.close()

	#--------------------------------------------------------
	#oneHotDict()
	#oneHotStatistics()
	#label_segmentation()
	#tfIdfDict()
	#pcaTrain()
	#ldaTrain()
	'''
	X, Y = sample(10)
	Z = tfIdfGet(Y)
	print(Y[0])
	for i in Z[0]:
		if i != 0:
			print(i)
	X, Y = sample(1000)
	Z = tfIdfGet(Y)
	ldaTrain(Z)
	pcaTrain(Z)
	X, Y = sample(5)
	Y = tfIdfGet(Y)
	print(ldaGet(Y))
	print(pcaGet(Y))
	'''

	return


if __name__ == '__main__':
	main()