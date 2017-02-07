import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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
	def stem(conversation):
		stemmer = PorterStemmer()
		conversation = [stemmer.stem(word) for word in conversation if not word in set(stopwords.words('english'))] # #removing all words in the stop word list
		conversation = ' '.join(conversation)
		return conversation
	@staticmethod
	def remove_tags(conversation_with_tags):
		return re.sub('<.*?>', ' ', conversation_with_tags)

	@staticmethod
	def remove_punctuation(conversation):
		return re.sub('[^a-zA-Z]', ' ', conversation)