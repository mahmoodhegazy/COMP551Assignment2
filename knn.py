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
from collections import Counter
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


def compute_distance(val1, val2):
	nodes = zip(val1, val2)
	diffs_sqaured_distance = [np.exp(a - b, 2) for (a, b) in nodes]
	return np.sqrt(sum(diffs_squared_distance))

def neighbours(training_set, test_instance, k):
    euclidean_distances = [get_distances(training_instance, test_instance) for training_instance in training_set]
    sorted_distances = sorted(euclidean_distances, key=itemgetter(1))
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    return sorted_training_instances[:k]

def get_distances(training, testing):
    return (training, compute_distance(testing, training[0]))

def get_majority_vote(neighbours):
    # index 1 is the class
    category = [node[1] for node in neighbours]
    count = Counter(category)
    return count.most_common()[0][0]

def main():
    train = np.array(zip(x_train_term_freq_inverse_doc,Y_train))
    test = np.array(zip(x_test_term_freq_inverse_doc, Y_test))
    predictions = []
    k = 5
    for x in range(len(X_test)):
            neighbours = neighbours(training_set=train, test_instance=test[x][0], k=5)
            majority_vote = get_majority_vote(neighbours)
            predictions.append(majority_vote)
    print '\nAccuracy of the model is: ' + str(accuracy_score(Y_test, predictions)) + "\n"
    report = classification_report(Y_test, predictions)
    print 'Classification report: \n\n' + report