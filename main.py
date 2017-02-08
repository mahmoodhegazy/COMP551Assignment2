from sanitize_data import sanitize_data
from SKLearn_Algorithms.decision_trees import decision_trees

X,Y = sanitize_data()
X = [x.encode('ascii') for x in X]
decision_trees(X, Y)