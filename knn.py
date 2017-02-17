import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from utilities import Utilities
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from operator import itemgetter
import operator
import math
from collections import Counter

def compute_distance(data1, data2):
	points = zip(data1, data2)
	diffs_sqaured_distance = [pow(a - b, 2) for (a, b) in points]
	return math.sqrt(sum(diffs_squared_distance))

def get_neighbours(training_set, test_instance, k):
    distances = [_get_tuple_distance(training_instance, test_instance) for training_instance in training_set]
    # index 1 is the calculated distance between training_instance and test_instance
    sorted_distances = sorted(distances, key=itemgetter(1))
    # extract only training instances
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    # select first k elements
    return sorted_training_instances[:k]

def _get_tuple_distance(training_instance, test_instance):
    return (training_instance, compute_distance(test_instance, training_instance[0]))

def get_majority_vote(neighbours):
    # index 1 is the class
    classes = [neighbour[1] for neighbour in neighbours]
    count = Counter(classes)
    return count.most_common()[0][0]

def main():
    # reformat train/test datasets for convenience
    train = np.array(zip(x_train_term_freq_inverse_doc,Y_train))
    test = np.array(zip(x_test_term_freq_inverse_doc, Y_test))
    # generate predictions
    predictions = []
    # let's arbitrarily set k equal to 5, meaning that to predict the class of new instances,
    k = 5
    # for each instance in the test set, get nearest neighbours and majority vote on predicted class
    for x in range(len(X_test)):
            print 'Classifying test instance number ' + str(x) + ":",
            neighbours = get_neighbours(training_set=train, test_instance=test[x][0], k=5)
            majority_vote = get_majority_vote(neighbours)
            predictions.append(majority_vote)
            print 'Predicted label=' + str(majority_vote) + ', Actual label=' + str(test[x][1])
    # summarize performance of the classification
    print '\nThe overall accuracy of the model is: ' + str(accuracy_score(Y_test, predictions)) + "\n"
    report = classification_report(Y_test, predictions)
    print 'A detailed classification report: \n\n' + report