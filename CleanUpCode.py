
# coding: utf-8

# In[38]:

import pandas as pd 
import numpy as np
from pandas import DataFrame, Series

left = pd.read_csv('/Users/praynaa/Downloads/train_input.csv')
right = pd.read_csv('/Users/praynaa/Downloads/train_output.csv')

frames = [left, right]
joined = pd.concat(frames, axis=1)
joined.shape


# In[4]:

category = np.where(joined['category'].str.contains('new'), 'Yes', 'No')


# In[5]:

category[:10]


# In[7]:

#or equivalently:
#j = pd.merge(left, right) # merges the two datasets

joined['conversation'][0]


# In[43]:

import re
import nltk
nltk.download('stopwords') #contains the list of irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from __future__ import print_function, unicode_literals

corpus = []

#Cleaning the texts
for i in range(0,10):
    notconversation = joined['conversation'][i]
    conversation = re.sub('<.*?>', ' ', notconversation) #removes all <*>
    conversation = re.sub('[^a-zA-Z]', ' ', conversation) #removes all punctuation
    conversation = conversation.lower() #sets everything to lowercase
    conversation = conversation.split()
    
    #Stemming (taking the root of the word)
    stemmer = PorterStemmer()
    conversation = [stemmer.stem(word) for word in conversation if not word in set(stopwords.words('english'))] #removing all words in the stop word list
    conversation = ' '.join(conversation)
    
    corpus.append(conversation)


#Stemming (taking the root of the word)
corpus = pd.DataFrame(corpus)
corpus


# In[ ]:



