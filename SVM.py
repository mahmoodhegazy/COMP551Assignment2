
# coding: utf-8

# In[1]:

import sklearn
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from math import isnan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm


# In[17]:

training_set = pd.read_csv("train_input.csv")


# In[18]:

training_set_array = np.array(training_set) #convert it to numpy array for manipulation


# In[19]:

training_set_array


# In[20]:

training_set_array = np.delete(training_set_array, [0], 1)  # remove headers


# In[21]:

training_set_array


# In[22]:

out_set = pd.read_csv('train_output.csv')


# In[23]:

out_set_array = np.array(out_set)


# In[24]:

out_set_array = np.delete(out_set_array, [0], 1)


# In[25]:

#create a scikit countvectorizer, this will do stemming and remove stop words as well as term frequency
count_vect = CountVectorizer(min_df = 2, lowercase = True, stop_words = 'english', strip_accents = 'unicode') 


# In[26]:

training_set_array = training_set_array.ravel()


# In[27]:

training_set_array


# In[28]:

inputWordCounter = count_vect.fit_transform(training_set_array) # feed the data to the vectorizor


# In[30]:

#create term frequency inverse document frequncy vector from our data
term_frequency_transformer = TfidfTransformer(use_idf=True).fit(inputWordCounter) 


# In[31]:

term_frequency_inverse_doc = term_frequency_transformer.transform(inputWordCounter)


# In[32]:

out_set_array = out_set_array.ravel()


# In[33]:

out_set_array


# In[43]:

test_set = pd.read_csv("test_input.csv")
test_set_array = np.array(test_set)
test_set_array = np.delete(test_set_array,[0],1) #remove fucking headers


# In[44]:

test_set_array = test_set_array.ravel()


# In[ ]:




# In[45]:

#feed it through the same vectorizing process we did to our training data
#transformed_test_count = count_vect.fit_transform(test_set_array)
test_counter = count_vect.transform(test_set_array) 


# In[46]:

test_set_frequency = term_frequency_transformer.transform(test_counter)


# In[48]:

model = svm.LinearSVC() #create a linear svm model


# In[50]:

model.fit(term_frequency_inverse_doc, out_set_array) #train it using our input and out training data


# In[51]:

predictions = model.predict(test_set_frequency)


# In[52]:

predictions


# In[55]:

to_submit = pd.DataFrame([[i,predictions[i]] for i in range(len(predictions))], columns = ["id","predicted category"])


# In[56]:

to_submit


# In[67]:

test_out_set = pd.read_csv("test_predict_random.csv")


# In[69]:

y_test = test_out_set.category.tolist()


# In[65]:

from sklearn.metrics import classification_report


# In[75]:

y_test


# In[ ]:




# In[78]:

predictions = predictions.tolist()


# In[81]:

print(classification_report(y_test,predictions))


# In[ ]:



