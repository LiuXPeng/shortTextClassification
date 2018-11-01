

f = open('test.txt', 'a', encoding = 'utf-8')
temp = [0] * 65000
for i in range(10000):
	f.write(' '.join(str(temp)) + '\n')

f.close()