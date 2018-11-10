#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'clean and static for twitter'

__author__ = 'lxp'

import time
import re
import nltk
import sys
import logging



#===================================按照天提取数据库中的数据================================
#-----------------------------------输入格式20181025-------------------------------
def extractData(time):
	if len(time) != 8:
		print('time is wrong')
		return

	dataBaseFilename = ''
	#这里日期格式需要处理一下
	if time[4] == '0':
		dataBaseFilename = time[:4] + time[5]
	else:
		dataBaseFilename = time[:6]
	dataBaseFilename = dataBaseFilename + 'twitter_data'
	client = pm.MongoClient()
	db = client.twitter
	tweets = db[dataBaseFilename]#"20187twitter_data"
	#------------把原有内容擦掉----------------
	file = open(time + "_twiter_data.txt","w",encoding='utf-8')
	file.truncate()
	file.close()
	#-----------------------------------------
	file = open(time + "_twiter_data.txt","a",encoding='utf-8')
	#日期要转换格式
	time1 = timeTrans(time[:4] + '-' + time[4:6] + '-' + time[6:] + ' 00:00:00') + '0000000000000000'
	time2 = timeTrans(time[:4] + '-' + time[4:6] + '-' + time[6:] + ' 23:59:59') + '0000000000000000'
	'''
	#all_tweets = tweets.find({"$and": [{"text": {"$regex": "SpaceX", "$options": "$i"}},
                                       {"text": {"$regex": "flacon", "$options": "$i"}}, {
                                           "_id": {"$gte": ObjectId("5bb0f3000000000000000000"),
                                                   "$lte": ObjectId("5bb244800000000000000000")}}]})
	'''
	all_tweets = tweets.find({"_id": {"$gte": ObjectId(time1[2:]), "$lte": ObjectId(time2[2:])}})
	#all_tweets = tweets.find({"$and": [{"text": {"$regex": "sparklab", "$options": "$i"}},
	#                                   {"text": {"$regex": "blockchain", "$options": "$i"}}, {
     #                                      "_id": {"$gte": ObjectId("5b97e6800000000000000000"),
      #                                             "$lte": ObjectId("5ba3c4000000000000000000")}}]})
    # cha zhao guan jian ci de  yu
    #db.insert_test.find({$and: [{"text":{$regex:"aaa"}},{"text":{$regex:"bbb"}}]}).count();
	count = 0
	for tweet in all_tweets:
		count = count + 1
		id = tweet['id_str']
		created_at = tweet['created_at']
		user_name = tweet['user']['name']
		user_screenname = tweet['user']['screen_name']
		user = user_name+"(@"+user_screenname+")"
		url = "https://twitter.com/" + str(user_screenname) + "/status/" + str(id)
		tags = []
		hashtags = []
		if('extended_tweet' in tweet.keys()):
			string = tweet['extended_tweet']['full_text'].replace('\n','')
			hashtags = tweet['entities']['hashtags']

		elif('retweeted_status' in tweet.keys()):
			if('extended_tweet' in tweet['retweeted_status'].keys()):
				string = tweet['retweeted_status']['extended_tweet']['full_text'].replace('\n','')
			else:
				string = tweet['retweeted_status']['text'].replace('\n','')
			hashtags = tweet['retweeted_status']['entities']['hashtags']
		else:
			string = tweet['text'].replace('\n','')


		string = string.replace('\r','')
		string_length = len(string)

		if(len(hashtags) != 0):
			for item in hashtags:
				tags.append(item['text'])
		time = str(created_at).replace('+0000','CDT')
		# file.write(time + '\t')
		file.write(id+"\t"+str(created_at)+"\t"+user_name+"(@"+user_screenname+")\t"+url+"\t"+str(tags)+"\t"+string + "\n")

        # file.write(str(tweet['user']['followers_count']) +"\n")


	file.close()
	return


#==========================时间格式转换========================
def timeTrans(a):
	timeArray = time.strptime(a,"%Y-%m-%d %H:%M:%S")
	timeStamp = int(time.mktime(timeArray))
	return str(hex(timeStamp))



#=======================================================================================================
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
                '#retweet', 'retweet', 'goooooooooo', 'hellooo', 'gooo', 'fucks', 'fuck', 'bitch', 'wey', 'sooo', 'helloooooo', 'lol', 'smfh', 'dont', 'lmaooo'])
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
	tag = False
	hashtag = ''
	features = []
	for i in range(len(text)):
		print(text[i])
		if text[i] == '#':
			tag = True
			continue
		if tag and text[i] != ' ':
			hashtag += text[i]
			print(hashtag)
		if tag and text[i] == ' ':
			tag = False
			features.append(hashtag)
			hashtag = ''


	tokens = []
	tokens = text.split()
	stop_words = load_stopwords()
	for word in tokens:
			if word.lower() not in stop_words:
				features.append(word)
	print
	return features

def tokenize(text):
	tag = False
	hashtag = ''
	features = []
	for i in range(len(text)):
		print(text[i])
		if text[i] == '#':
			tag = True
			continue
		if tag and text[i] != ' ':
			hashtag += text[i]
			print(hashtag)
		if tag and text[i] == ' ':
			tag = False
			features.append(hashtag)
			hashtag = ''

	text = normalize_text(text)
	words = nltk_tokenize(text)
	words += features
	return words

print(tokenize('fasdlk #dfdsjlfak dfjakslfj dffa'))