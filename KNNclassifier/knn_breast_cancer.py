'''
K Nearest Neighbors Algorithm using sklearn
based on tutorial by sentdex
using uci breast cancer dataset
does not scale well, bad on large datasets
@TODO: learn neural networks lol
ISO
'''




import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


'''
preprocessing is the module used to do some cleaning/scaling of data prior to machine learning
cross_ alidation is used in the testing stages
'''
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# replace unkown data with outliers
df.replace('?',-99999, inplace=True)
# irrelevent feature
df.drop(['id'], 1, inplace=True)

# feature data
X = np.array(df.drop(['class'], 1))
# class / label data
y = np.array(df['class'])

# separate training and testing chunks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define classifier
clf = neighbors.KNeighborsClassifier()

# train classifier
clf.fit(X_train, y_train)

# test
accuracy = clf.score(X_test, y_test)
print(accuracy)
# about 95% accuracy without any tweaks
# If you want to save the classier you'd pickle it

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
