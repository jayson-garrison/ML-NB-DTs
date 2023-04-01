import pandas as pd
from sklearn import tree
import numpy as np
from Utils import five_fold as ff

data = pd.read_csv("abalone.data")
data = data.to_numpy()

fold = ff.five_fold(data.tolist())
X_tr = np.asarray(fold[0][0])[1:, 1:]
Y_tr = np.asarray(fold[0][0])[1:,0]
X_te = np.asarray(fold[0][1])[1:, 1:]
Y_te = np.asarray(fold[0][1])[1:,0]

print(len(X_tr))
print(len(Y_tr))
print(len(X_te))
print(len(Y_te))

'''
create a basic decision tree from the tree library in scikit learn
we select the entropy method of calculating impurity

the library uses Shannon Information Gain to determine which attribute to split on
which is effectively the same thing as minimizing the log loss. The log loss is the 
same as the entropy formula that we use in class.

minimizing the log loss is effectively what we do when calculating the mutual information
gain in deciding on what attribute to split on. if we minimize the log loss, that is to say that
we minimize entropy calculation given we  split on a certain attribute A, (this is H(Y|X)), we are
maximzizing the information gain as IG(Y|X) = H(X) - H(Y|X).

the fit
'''

dt = tree.DecisionTreeClassifier(criterion="entropy")
dt.fit(X_tr, Y_tr)

#tree.plot_tree(dt)

# make some unseen examples and see what the test accuracy is:

predictions = dt.predict(X_te)

#print(predictions)

# find the test accuracy

accuracy = 0
for idx, prediction in enumerate(predictions):
    if (prediction == Y_te[idx]):
        accuracy += 1
accuracy = accuracy / len(Y_te)

print('Test accuracy: ', accuracy)