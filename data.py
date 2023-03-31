import pandas as pd
from sklearn import tree
import numpy as np

data = pd.read_csv("agaricus-lepiota.data")
data = data.to_numpy()
X = data[0:, 1:]
Y = data[0:, 0]