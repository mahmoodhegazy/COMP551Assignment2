import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re

nltk.download('stopwords') #list of irrelevant words

class Utilities():
	@staticmethod
	def convert_category(x):
		if (x == 'hockey'):
			return 0
		elif (x == 'movies'):
			return 1
		elif (x == 'nba'):
			return 2
		elif (x == 'news'):
			return 3
		elif (x == 'nfl'):
			return 4
		elif (x == 'politics'):
			return 5
		elif (x == 'soccer'):
			return 6
		elif (x == 'worldnews'):
			return 7
		else:
			return -1 #sentinel value for invalid category

	@staticmethod
	def lemmatize(conversation):
		wordnet_lemmatizer = WordNetLemmatizer()
		conversation = [wordnet_lemmatizer.lemmatize(word) for word in conversation if not word in set(stopwords.words('english'))]
		conversation = ' '.join(conversation)
		return conversation 

	@staticmethod
	def remove_tags(conversation_with_tags):
		return re.sub('<.*?>', ' ', conversation_with_tags)

	@staticmethod
	def remove_punctuation(conversation):
		return re.sub('[^a-zA-Z]', ' ', conversation)

	@staticmethod
	def read_input_data():
		training_set_input = pd.read_csv('./train_data/train_input.csv')
		training_set_output = pd.read_csv('./train_data/train_output.csv')
		training_set_input_no_header = np.delete(np.array(training_set_input), [0], 1).ravel()
		training_set_output_no_header = np.delete(np.array(training_set_output), [0], 1).ravel()
		return [training_set_input_no_header, training_set_output_no_header]

	@staticmethod
	def stem(conversation):
		stemmer = PorterStemmer()
		conversation = [stemmer.stem(word) for word in conversation if not word in set(stopwords.words('english'))]
		conversation = ' '.join(conversation)
		return conversation

	@staticmethod
	def write_to_csv(corpus, frame):
		df_conversation = pd.DataFrame(corpus)  
		df_conversation = df_conversation.rename(columns={0: 'conversation'})
		df_categories = pd.DataFrame(right['category'])
		df_categories_test = df_categories.head(2000)
		concat_frames = [df_conversation, df_categories_test]
		joined_frames = pd.concat(concat_frames, axis=1)
		joined_frames.to_csv('./cleaned_data/CLEANED_DATA.csv', index = False)
