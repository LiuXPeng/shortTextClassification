#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'去除重复行'

__author__ = 'lxp'

def test():
	shuruwenjian = input('请输入要去重的文件名： ')
	shuchuwenjian = input('请输入要输出的文件名： ')
	f = open(shuruwenjian, 'r', encoding='utf-8')
	line = f.readline()
	g = open(shuchuwenjian, 'w+')#之所以分词和类别分开，是为了向量化训练方便
	g.truncate()#清空文件，防止测试时候不停加入
	g = open(shuchuwenjian, 'a', encoding='utf-8')
	bag = set()
	n = 0
	m = 0
	while line:
		n = n + 1
		if line in bag:
			line = f.readline()
			m = m + 1
			continue
		bag.add(line)
		g.write(line)
		line = f.readline()
	g.close()
	f.close()
	print('n =', n)
	print('m =', m)
	print("去重率：", m / n)
	return

if __name__ == '__main__':
	test()