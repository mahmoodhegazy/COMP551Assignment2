
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



df_clean = pd.read_csv('full_sanitized_data.csv')
cv = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1,3))
word_counter = cv.fit_transform(df_clean.clean_sentence.tolist())

X_train_grid, X_test_grid, Y_train_grid, Y_test_grid = train_test_split(word_counter, df_clean.category, test_size = 0.30, random_state = 101)
# GridSearchCV is a meta-classifier. It takes an classifier like SVC, and creates a new classifier, that behaves exactly the same
# its used to help us find the right parameters for SVM (to get the best predcition)
# we can be a little lazy and just try a bunch of combinations and see what works best
param_grid = {'C': [10], 'gamma': [0.001,0.0001], 'kernel': ['rbf']} # Parameter I want to try out for SVM
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train_grid,Y_train_grid) # will proabably take a while :/
 # Now you can re-run predictions on this grid object just like you would with a normal model.
grid_predictions = grid.predict(X_test_grid)
conf_matrix_grid = confusion_matrix(Y_test_grid,grid_predictions)
#    print(confusion_matrix(y_test,grid_predictions))
#    print(classification_report(y_test,grid_predictions))
print(classification_report(Y_test,Y_pred))
get_ipython().magic(u'matplotlib inline')
sns.heatmap(confusion_mat,annot=True)



