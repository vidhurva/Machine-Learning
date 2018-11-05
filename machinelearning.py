#import scikit-learn machine learning software
from sklearn import tree

#create semi-realistic training data
features = [[7.0, 1], [7.8, 1], [6.7, 2], [5.2, 3], [6.3, 2], [5.6, 3], [2.5, 4], [3.2, 5], [8.0, 1], [6.5, 2], [2.4, 4], [3.1, 5]]
drink = ["water", "water", "vodka", "gin", "vodka", "gin", "pepsi", "rosé", "water", "vodka", "pepsi", "rosé"]

#train classifier --> Decision Tree to find patterns in data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,drink)

#create predictions based on training data
prediction = clf.predict([[3.5, 5]])
prediction2 = clf.predict([[7.0, 5]])

print(prediction)
print(prediction2)
