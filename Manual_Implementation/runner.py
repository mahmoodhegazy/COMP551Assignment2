import csv
import pandas as pd
import numpy as np
import re
import nltk
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pprint import pprint
from scipy.sparse import csr_matrix
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from collections import defaultdict, Counter
from collections import namedtuple
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import MultinomialNaiveBayes

#This files runs the runner and calls the Multinomial Naive Bayes File

labels = ["hockey", "movies", "nba", "news", "nfl", "politics", "soccer", "worldnews"]



def parse_classification(datafile): # dictionary of each category and the docID pertaining to it 
		'''
		USE FOR _OUTPUT FILES
		function that goes through the given datafile and creates a dictionary of
		{document: classification}
		@params string datafile: the datafile to be opened
		@return dict classification: the classification of documents parsed
		'''

		data = []
		with open(datafile,'rb') as csvfile:
			reader = csv.reader(csvfile)
			next(reader)
			for row in reader:
				data.append(row)
		dictionary = {}
		for line in data:
			line[0] = int(line[0])
			if line[1] in dictionary:
				dictionary[line[1]].append(line[0])
			else:
				dictionary[line[1]] = [line[0]]

		return dictionary


X_train = pd.read_csv('X_train_input.csv')
X_test = pd.read_csv('X_test_input.csv')
y_test = pd.read_csv('y_test_output.csv')

X_train.drop(X_train[[0]], axis=1, inplace=True)
X_test.drop(X_test[[0]], axis=1, inplace=True)
y_test.drop(y_test[[0]], axis=1, inplace=True)


y_train_dictionary = parse_classification('y_train_output.csv')
y_test = np.array(y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)
classifier = MultinomialNaiveBayes.Multinomial_Naive_Bayes()
prior, vocab, cond_prob_cache = classifier.train_MNB_model(labels, X_train, y_train_dictionary)
y_pred = []
for document in X_test:
	prediction = classifier.predict_with_MNB_classifier(labels, vocab, prior, cond_prob_cache, document[0])
	y_pred.append(prediction)

y_pred = np.array(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



