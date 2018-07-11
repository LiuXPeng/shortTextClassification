#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'数据抽样'

__author__ = 'lxp'

def test():
	shuruwenjian = input('请输入要抽样的文件名： ')
	shuchuwenjian = input('请输入要输出的文件名： ')
	k = int(input('抽样间隔'))
	f = open(shuruwenjian, 'r', encoding='utf-8')
	line = f.readline()
	g = open(shuchuwenjian, 'w+')
	g.truncate()#清空文件，防止测试时候不停加入
	g = open(shuchuwenjian, 'a', encoding='utf-8')
	n = 0
	m = 0
	while line:
		n = n + 1
		if n % k != 0:
			line = f.readline()
			continue
		m = m + 1
		g.write(line)
		line = f.readline()
	g.close()
	f.close()
	print('抽取数据总数：', m)
	return

if __name__ == '__main__':
	test()