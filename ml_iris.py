from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
from sklearn import tree
import numpy as np
import graphviz
import pydotplus
from IPython.display import Image

iris = load_iris()
test_idx = [0, 50, 100] # first of every target

for feature in iris.feature_names:
    print ("Feature: " + feature)
    pass

for target in iris.target_names:
    print ("Target: " + target)
    pass

for i in range(len(iris.target)):
    print ("Example %d: label: %s, feature: %s" % (i, iris.target_names[iris.target[i]], iris.data[i]))
    pass

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# train
clf = tree.DecisionTreeClassifier()
fit = clf.fit(train_data, train_target)

# predict
print (test_target)
print (clf.predict(test_data))

# visualize
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, 
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("iris.png")
