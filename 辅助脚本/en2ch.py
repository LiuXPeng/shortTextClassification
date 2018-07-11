#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'english text translate to chinese'

__author__ = 'lxp'


import time
import translate

def test():
	shuruwenjian = input('请输入要翻译的文件名： ')
	shuchuwenjian = input('请输入要输出的文件名： ')
	s = input('是否清楚现有输出文件的内容,输入y表示同意')
	if s != 'y':
		return
	f = open(shuruwenjian, 'r', encoding='utf-8')
	line = f.readline()
	g = open('samples_chinese_better.txt', 'w+')
	g.truncate()#清空文件，防止测试时候不停加入
	g.close()
	g = open(shuchuwenjian, 'a', encoding='utf-8')
	while line:
		t = translate.baidu_translate(line)
		print(str(t))
		g.write(str(t))
		g.write('	')
		g.write(line)
		line = f.readline()
		time.sleep(1)
	g.close()
	f.close()
	return

if __name__ == '__main__':
	test()
