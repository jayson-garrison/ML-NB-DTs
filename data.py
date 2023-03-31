import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from naive_bayes import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#X, y = datasets.make_classification(n_samples=1000, n_features=9, n_classes = 2, random_state=123)

data = pd.read_csv("abalone.data")
data = data.to_numpy()
X = data[0:, 1:]
y = data[0:, 0]

for idx, label in np.ndenumerate(y):
    if (label == 'M') :
        y[idx] = 0.333
    elif (label == 'F'):
        y[idx] = 0.677
    elif (label == 'I'):
        y[idx] = 1

y_int = np.array(y, dtype=int)
print(y_int)


X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.20, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))