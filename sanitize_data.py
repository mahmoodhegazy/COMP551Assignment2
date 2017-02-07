import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from utilities import Utilities

util = Utilities()

left = pd.read_csv('./train/train_input.csv')
right = pd.read_csv('./train/train_output.csv')

frames = [left, right['category']] #create columns id, conversation, category
joined = pd.concat(frames, axis=1)

corpus = []

#Cleaning the texts
for i in range(0,2000):
    conversation_with_tags = joined['conversation'][i]
    conversation = util.remove_tags(conversation_with_tags)
    conversation = util.remove_punctuation(conversation) 
    conversation = conversation.lower().split() #sets everything to lowercase and splits on the spaces by default
    conversation = util.stem(conversation) #stemming (taking the root of the word)
    corpus.append(conversation)

df_conversation = pd.DataFrame(corpus)  
df_conversation = df_conversation.rename(columns={0: 'conversation'})
df_categories = pd.DataFrame(right['category'])
df_categories_test = df_categories.head(2000)
concat_frames = [df_conversation, df_categories_test]
joined_frames = pd.concat(concat_frames, axis=1)

joined_frames.to_csv('./cleaned_data/CLEANED_DATA.csv', index = False)