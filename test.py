import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

sent = " for use with Digital Computers.\n\nby W. N. Francis and H. Kucera (1964)\nDepartment of Linguistics, Brown University\nProvidence, Rhode Island, USA\n\nRevised 1971, Revised and Amplified 1979\n\nhttp://www.hit.uib.no/icame/brown/bcm.html\n\nDistributed with the permission of"
words = word_tokenize(sent)

words = [[word for word in document if not word in english_stopwords] for document in words]

print(words)