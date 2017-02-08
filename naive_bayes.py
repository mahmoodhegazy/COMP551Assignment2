from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word', ngram_range=(1,3), max_features=1500) #Chooses the top 1500 features
X = cv.fit_transform(corpus).toarray()

y = right.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Fitting Naive Bayes to Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmnb = confusion_matrix(y_test, y_pred)

print(classifier.score(X_test, y_test))