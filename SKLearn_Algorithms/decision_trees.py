from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def decision_trees(X, Y):
    confusion_mat = 1
    cv = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1,3), max_features=1500)
    word_counter = cv.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(word_counter, Y, test_size = 0.30, random_state = 0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    print('Score: ')
    print(clf.score(X_test, Y_test))