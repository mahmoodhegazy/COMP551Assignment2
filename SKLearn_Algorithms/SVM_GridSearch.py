
# coding: utf-8

# In[168]:

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from utilities import Utilities
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[145]:

#ignore terms that appear in less than 2 documents
#ignore terms that appear in more than 40% of the data
cv = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1,3),stop_words='english', strip_accents = 'unicode',min_df = 2)
df_full_input = pd.read_csv('cleaned_data/full_input_space_forcv.csv')
word_counter = cv.fit_transform(df_full_input.clean_sentence.tolist()) # sparse input (for both training and testing data)

word_counter_input_test = word_counter[165000:218218] # last 53218 entries are for kaggle testing
word_counter_input_train = word_counter[0:165000] #take all non-kaggle data
# cross validations : 70/30 train/test split
df_clean = pd.read_csv('train_data/full_sanitized_train_data.csv') #clean trainig data set
X_train, X_test, Y_train, Y_test = train_test_split(word_counter_input_train, df_clean.category, test_size = 0.30, random_state = 101)
#create term frequency inverse document frequncy vectors from the data
x_train_term_freq_transformer = TfidfTransformer(use_idf=True).fit(X_train) 
x_test_term_freq_transformer = TfidfTransformer(use_idf=True).fit(X_test) 
kaggle_xtest_term_freq_transformer = TfidfTransformer(use_idf=True).fit(word_counter_input_test) 
x_train_term_freq_inverse_doc = x_train_term_freq_transformer.transform(X_train)
x_test_term_freq_inverse_doc = x_test_term_freq_transformer.transform(X_test)
kaggle_xtest_term_freq_inverse_doc = kaggle_xtest_term_freq_transformer.transform(word_counter_input_test)
# GridSearchCV is a meta-classifier. It takes an classifier like SVC, and creates a new classifier, that behaves exactly the same
# its used to help us find the right parameters for SVM (to get the best predcition)
# we can be a little lazy and just try a bunch of combinations and see what works best
param_grid = {'C': [10], 'tol': [0.9] ,  } # Parameter I want to try out for SVM
grid = GridSearchCV(svm.LinearSVC(),param_grid , refit=True,verbose=3)
grid.fit(x_train_term_freq_inverse_doc, Y_train)
grid_predictions = grid.predict(x_test_term_freq_inverse_doc)
print(classification_report(Y_test,grid_predictions))
# confusion matrix
confusion_mat = confusion_matrix(Y_test, grid_predictions)
get_ipython().magic(u'matplotlib inline')
sns.heatmap(confusion_mat,annot=True) # conf matrix (for report)


y_kaggle_pred = grid.predict(kaggle_xtest_term_freq_inverse_doc)
to_submit = pd.DataFrame([[i,y_kaggle_pred[i]] for i in range(len(y_kaggle_pred))], columns = ["id","category"]) #follow submission guidelines 
to_submit.to_csv("kaggle_prediciton_submissions/submission_9_svm.csv", index= False) #kaggle

print(grid.best_params_)
print(accuracy_score(Y_test, grid_predictions))




