import numpy as np
#import graphviz
from sklearn.datasets import load_iris
iris = load_iris()
test_idx = [0,50,100]
from sklearn import tree

#metadata --> data already provided by scikitlearn
print(iris.feature_names)
print(iris.target_names)

#training data (contains majority of data)
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data (tests how well the machine can predict new answers from training)
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#implementing decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
prediction = clf.predict(test_data)
print(prediction)
