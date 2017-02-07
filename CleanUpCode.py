
# coding: utf-8

# In[38]:

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import re
import nltk
nltk.download('stopwords') #contains the list of irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

left = pd.read_csv('/Users/praynaa/Downloads/train_input.csv')
right = pd.read_csv('/Users/praynaa/Downloads/train_output.csv')


frames = [left, right]
joined = pd.concat(frames, axis=1)


category = np.where(joined['category'].str.contains('new'), 'Yes', 'No')


corpus = []
i = 0

def convert(x):
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
        return 999999

right['category'] = right.category.apply(convert)

#Cleaning the texts
for i in range(0,165000):
    notconversation = joined['conversation'][i]
    conversation = re.sub('<.*?>', ' ', notconversation) #removes all <*>
    conversation = re.sub('[^a-zA-Z]', ' ', conversation) #removes all punctuation
    conversation = conversation.lower() #sets everything to lowercase
    conversation = conversation.split()
    
    #Stemming (taking the root of the word)
    stemmer = PorterStemmer()
    conversation = [stemmer.stem(word) for word in conversation if not word in set(stopwords.words('english'))] #removing all words in the stop word list
    conversation = ' '.join(conversation)
    
    i = i+1
    if(i%100 == 0):
        print(i)
    corpus.append(conversation)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) #Chooses the top 1500 features
X = cv.fit_transform(corpus).toarray()

y = right.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Fitting Naive Bayes to Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)