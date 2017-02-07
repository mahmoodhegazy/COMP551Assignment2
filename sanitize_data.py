import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from multinomial_nb import naive_bayes
from utilities import Utilities

util = Utilities()

[left, right] = util.read_input_data()

frames = [left, right['category']] #create columns id, conversation, category
joined = pd.concat(frames, axis=1)

corpus = []

#Cleaning the texts
for i in range(0,10000):
    conversation_with_tags = joined['conversation'][i]
    conversation = util.remove_tags(conversation_with_tags)
    conversation = util.remove_punctuation(conversation) 
    conversation = conversation.lower().split() #sets everything to lowercase and splits on the spaces by default
    #conversation = util.stem(conversation) #stemming (taking the root of the word)
    conversation = util.lemmatize(conversation)
    corpus.append(conversation)

naive_bayes(corpus, right)