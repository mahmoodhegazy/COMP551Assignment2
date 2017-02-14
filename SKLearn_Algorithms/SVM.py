
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



cv = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1,3),max_features=2836998)  #max out the number of features constructed to the number of features the kaggle input can construct 
df_full_input = pd.read_csv('../cleaned_data/full_input_space_forcv.csv')
df_clean = pd.read_csv('../train_data/full_sanitized_train_data.csv') #clean trainig data set
word_counter = cv.fit_transform(df_full_input.clean_sentence.tolist()) # sparse input (for both training and testing data)
word_counter_input_test = word_counter[165000:218218] # last 53218 entries are for kaggle testing
word_counter_input_train = word_counter[0:165000] #take all non-kaggle data for training/testing split
# cross validations : 80/20 train/test split
X_train, X_test, Y_train, Y_test = train_test_split(word_counter_input_train, df_clean.category, test_size = 0.20, random_state = 101)
model = SVC(C=10,gamma=0.001,kernel='rbf')  #using best parameters found by grid search 
model = model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
confusion_mat = confusion_matrix(Y_test, y_pred)

print(classification_report(Y_test,y_pred)) # gave us 95% on train/test spplit (should expect same for kaggle test)

# now get kaggle preictions
y_kaggle_pred = model.predict(word_counter_input_test)
to_submit = pd.DataFrame([[i,y_kaggle_pred[i]] for i in range(len(y_kaggle_pred))], columns = ["id","category"]) #follow submission guidelines 
to_submit.to_csv("submission_svm.csv", index= False) #1st submission :)

sns.heatmap(confusion_mat,annot=True) # conf matrix (for report)


