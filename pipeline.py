import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# my_classifier = tree.DecisionTreeClassifier()
my_classifier = neighbors.KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))
