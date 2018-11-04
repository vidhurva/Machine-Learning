#import a library that lessens code for machine learning program
import sklearn
from sklearn import tree

#training data
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = ["apple", "apple", "orange", "orange"]

#train classifier --> Decision Tree to find patterns in data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
prediction = clf.predict([[150, 0]])

print(prediction)
