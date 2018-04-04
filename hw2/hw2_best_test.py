import sys
import math
import pickle
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression

test = pd.read_csv(sys.argv[1])

test_X = test.values

upup = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
for feature in upup:
	for times in np.arange(2, 100, 1):
		test_X = np.concatenate([test_X, (test[feature].values.reshape(-1, 1)**times)], axis=1)



### normalize ###
def normalize(x):
	mean = np.load("./best_mean.npy")
	dev = np.load("./best_dev.npy")
	x = (x-mean) / (dev)
	return x 

test_X = normalize(test_X)

with open('./model.pickle', 'rb') as f:
	model = pickle.load(f)
	y_pred = model.predict(test_X).reshape(-1, 1)
	label = np.arange(y_pred.shape[0]).reshape(-1, 1)
	label = label + 1
	ans = np.concatenate([label, y_pred], axis=1).astype(int)
	np.savetxt(sys.argv[2], ans, fmt="%s", header='id,label', comments='', delimiter=",")