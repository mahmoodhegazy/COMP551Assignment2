import csv
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pprint import pprint
from scipy.sparse import csr_matrix
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from collections import defaultdict, Counter
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from collections import namedtuple
import math
from itertools import islice
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
##################
#
#
#Class for the Multinomial Naive Bayes Classifier
#create_BOW: Creates the Bag Of Words for the classifier. Produces a big sparse matrix
#create_sparse_matricies: Returns a dictionary with a sparse matrix for each of the 8 categories
#extract_token_from_documents: Returns a list with all the tokens in a conversation/document
#train_MNB_model: Trains the MNB model and produces the dictionary with all the probability estimates P(t|c)
#predict_with_MNB_model: Does the predictions with the classifier that was trained
#
######################


#Tuple object required to store the probabilities of word and class/label in the dictionary later
cond_prob_tuple= namedtuple("cond_prob_tuple",["word", "label"])

class Multinomial_Naive_Bayes:

	def __init__(self):
		return


	def extract_tokens_from_documents(self, set_vocab, document):
		tokens = str.lower(document)
		tokens = tokens.split()
		clean_token = []
		for token in tokens:
			if token in set_vocab:
				clean_token.append(token) 
		return clean_token

	def create_BOW(self, parsed_data):
		count_sparse_matrix = []
		count_vectorizer = CountVectorizer(analyzer = "word",   
	                             tokenizer = None,    
	                             preprocessor = None, 
	                             stop_words = "english", 
	                             max_features = 25000)

		count_sparse_matrix = count_vectorizer.fit_transform(parsed_data.ravel())
		count_sparse_matrix = count_sparse_matrix.toarray()
		vocab = count_vectorizer.get_feature_names()
		return count_sparse_matrix, vocab

	def create_sparse_matricies(self, original_sparse_matrix, docs_in_class):
		sparse_matricies = {}
		for index_label, label in enumerate(docs_in_class):
			doc_numbers_per_class = docs_in_class.get(label)
			new_matrix = []
			for doc_number in doc_numbers_per_class:
				new_matrix.append(original_sparse_matrix[doc_number])
			new_matrix = np.array(new_matrix)
			sparse_matricies[label] = new_matrix
		return sparse_matricies

	def train_MNB_model(self, labels, documents, dictionary):
		prior = {}
		text = ""	
		count_features, vocab = self.create_BOW(documents)
		#print features
		all_sparse_matrix = self.create_sparse_matricies(count_features, dictionary)
		count = len(documents)
		vocab_count = len(vocab)
		#print vocab_count
		cond_prob_cache = {}
		for label in labels:
			num_of_docs_class = len(dictionary[label])

			prior[label] = float(num_of_docs_class)/float(count)
			label_matrix_sparse = all_sparse_matrix.get(label) 
			total_text_len = np.sum(label_matrix_sparse)
			sum_of_features = label_matrix_sparse.sum(axis=0)
			for index_word, word in enumerate(vocab):
				num_of_tokens_term = sum_of_features[index_word]
				cond_prob = float(num_of_tokens_term + 1)/float(total_text_len + vocab_count)
				#print temp_prob
				term_tuple = cond_prob_tuple(word = word, label = label)
				cond_prob_cache[term_tuple] = cond_prob

		return prior, vocab, cond_prob_cache

	def predict_with_MNB_classifier(self, labels, vocab, prior, cond_prob_cache, document):
		set_vocab = set(vocab)
		score = {}
		#print set_vocab
		document_tokens = self.extract_tokens_from_documents(set_vocab, document)
		for label in labels: 
			score[label] = float(math.log(prior.get(label)))
			for token in document_tokens:
				temp = cond_prob_cache.get(cond_prob_tuple(word=token, label=label))

				score[label] = float(score.get(label)) + float(math.log(temp))
		current = float('-inf')
		highest = float('-inf')
		maxLabel = "" 
		for key in score: 
			current = score.get(key)
			if current > highest: 
				highest = current
				maxLabel = key
		return maxLabel
