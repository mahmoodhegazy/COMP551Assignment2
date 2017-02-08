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
from sklearn.metrics import classification_report

training_set = pd.read_csv("train_input.csv")
training_set_array = np.array(training_set) #convert it to numpy array for manipulation
training_set_array
training_set_array = np.delete(training_set_array, [0], 1)  # remove headers
training_set_array

out_set = pd.read_csv('train_output.csv')
out_set_array = np.array(out_set)
out_set_array = np.delete(out_set_array, [0], 1)

#create a scikit countvectorizer, this will do stemming and remove stop words as well as term frequency
count_vect = CountVectorizer(min_df = 2, lowercase = True, stop_words = 'english', strip_accents = 'unicode') 
training_set_array = training_set_array.ravel()
training_set_array
inputWordCounter = count_vect.fit_transform(training_set_array) # feed the data to the vectorizor

#create term frequency inverse document frequncy vector from our data
term_frequency_transformer = TfidfTransformer(use_idf=True).fit(inputWordCounter) 
term_frequency_inverse_doc = term_frequency_transformer.transform(inputWordCounter)
out_set_array = out_set_array.ravel()
out_set_array

test_set = pd.read_csv("test_input.csv")
test_set_array = np.array(test_set)
test_set_array = np.delete(test_set_array,[0],1) #remove fucking headers
test_set_array = test_set_array.ravel()

#feed it through the same vectorizing process we did to our training data
#transformed_test_count = count_vect.fit_transform(test_set_array)
test_counter = count_vect.transform(test_set_array) 
test_set_frequency = term_frequency_transformer.transform(test_counter)
model = svm.LinearSVC() #create a linear svm model
model.fit(term_frequency_inverse_doc, out_set_array) #train it using our input and out training data
predictions = model.predict(test_set_frequency)
predictions
to_submit = pd.DataFrame([[i,predictions[i]] for i in range(len(predictions))], columns = ["id","predicted category"])
to_submit
test_out_set = pd.read_csv("test_predict_random.csv")
y_test = test_out_set.category.tolist()

y_test

predictions = predictions.tolist()
print(classification_report(y_test,predictions))