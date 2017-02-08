import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from utilities import Utilities

util = Utilities()
[training_set_X, training_set_Y] = util.read_input_data()

X = []
Y = training_set_Y[0:100]

def sanitize_data():
	for i in range(0,100):
		conversation_with_tags = training_set_X[i]
		conversation = util.remove_tags(conversation_with_tags)
		conversation = util.remove_punctuation(conversation) 
		conversation = conversation.lower().split() #sets everything to lowercase and splits on the spaces by default
		#conversation = util.stem(conversation) #stemming (taking the root of the word)
		conversation = util.lemmatize(conversation)
		X.append(conversation)
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

