from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer

f = open('label_segmentation.txt', 'r', encoding = 'utf-8')
line = f.readline()
tag_list = []
while line:
	L = line.split
	tag_list.append(L[1])

vectorizer = CountVectorizer() #将文本中的词语转换为词频矩阵  
X = vectorizer.fit_transform(tag_list) #计算个词语出现的次数
"""
word_dict = vectorizer.vocabulary_
{'唱歌': 2, '吃货': 1, '青年': 6, '足球': 5, '叛逆': 0, '少年': 3, '游戏': 4}
"""

transformer = TfidfTransformer()  
tfidf = transformer.fit_transform(X)  #将词频矩阵X统计成TF-IDF值  
print(tfidf.toarray())