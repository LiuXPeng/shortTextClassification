#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'svm for short text classification'

__author__ = 'lxp'

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence



def makeWordBag():
	f = open('segmentation.txt', 'r', encoding='utf-8')
	g = open('sentences.txt', 'w+')
	g.truncate()#清空文件，防止测试时候不停加入
	g.close()
	g = open('sentences.txt', 'a', encoding='utf-8')
	line = f.readline()
	while line:
		L = line.split(',', 1)
		line = f.readline()
		if L[0][0] == '1':
			g.write(L[1])
	f.close()
	f = open('segmentation.txt', 'r', encoding='utf-8')
	line = f.readline()
	while line:
		L = line.split(',', 1)
		line = f.readline()
		if L[0][0] == '-':
			g.write(L[1])	
	g.close()
	return


def w2v():
	sentences = LineSentence('sentences.txt')
	model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=4)

	print(model.wv['computer'])

#makeWordBag()
w2v()