# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:29:50 2020

@author: Snigdhabose
"""


from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analysis = TextBlob("TextBlob sure looks like it has some interesting features!")
#print(dir(analysis))
#print(analysis.translate(to='es'))
#print(analysis.tags)
print(analysis.sentiment)

    
pos_count = 0
pos_correct = 0

fp = open('C:\Users\Snigdhabose\Desktop\major proj\code\positive.txt', 'rb')

for line in fp.read().split('\n'):
        analysis = TextBlob(line)
        if analysis.sentiment.polarity > 0:
            pos_correct += 1
        pos_count +=1