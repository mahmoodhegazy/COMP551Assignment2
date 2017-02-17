
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from utilities import Utilities
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier  

#ignore terms that appear in less than 2 docs
cv = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1,3),stop_words='english', strip_accents = 'unicode',min_df = 2)
df_full_input = pd.read_csv('cleaned_data/full_input_space_forcv.csv')
word_counter = cv.fit_transform(df_full_input.clean_sentence.tolist()) # sparse input (for both training and testing data)
word_counter_input_test = word_counter[165000:218218] # last 53218 entries are for kaggle testing
word_counter_input_train = word_counter[0:165000] #take all non-kaggle data
# cross validations : 80/20 train/test split
df_clean = pd.read_csv('train_data/full_sanitized_train_data.csv') #clean trainig data set
X_train, X_test, Y_train, Y_test = train_test_split(word_counter_input_train, df_clean.category, test_size = 0.20, random_state = 101)
#create term frequency inverse document frequncy vectors from the data
x_train_term_freq_transformer = TfidfTransformer(use_idf=True).fit(X_train) 
x_test_term_freq_transformer = TfidfTransformer(use_idf=True).fit(X_test) 
kaggle_xtest_term_freq_transformer = TfidfTransformer(use_idf=True).fit(word_counter_input_test) 
x_train_term_freq_inverse_doc = x_train_term_freq_transformer.transform(X_train)
x_test_term_freq_inverse_doc = x_test_term_freq_transformer.transform(X_test)


kaggle_xtest_term_freq_inverse_doc = kaggle_xtest_term_freq_transformer.transform(word_counter_input_test)
neigh = KNeighborsClassifier(n_neighbors = 3)

neigh.fit(x_train_term_freq_inverse_doc, Y_train) #train it using our input and out training data
predictions = neigh.predict(x_test_term_freq_inverse_doc) #create a prediction for our testing data

print(classification_report(Y_test,predictions))
y_kaggle_pred = neigh.predict(kaggle_xtest_term_freq_inverse_doc)
to_submit = pd.DataFrame([[i,y_kaggle_pred[i]] for i in range(len(y_kaggle_pred))], columns = ["id","category"]) #follow submission guidelines 
to_submit.to_csv("kaggle_prediciton_submissions/submission_knn.csv", index= False) #1st submission :)

confusion_mat = confusion_matrix(Y_test, predictions)

get_ipython().magic(u'matplotlib inline')
sns.heatmap(confusion_mat,annot=True) # conf matrix (for report)

