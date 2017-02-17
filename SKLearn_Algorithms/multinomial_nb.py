from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

def naive_bayes(corpus, categories):
	cv = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(1,3))
	X = cv.fit_transform(corpus).toarray()
	y = categories.iloc[:,1].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
	classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	print('Score: ')
	print(classifier.score(X_test, y_test))