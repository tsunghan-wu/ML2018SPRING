import sys
import math
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
'''
[hw2 best method]
- power some column
- normalize
- sklearn logistic regression
- regularization
'''
data = {
	'trainX': pd.read_csv(sys.argv[1]),
	'trainY': pd.read_csv(sys.argv[2], names=['y']),
	'test_X': pd.read_csv(sys.argv[3]) 
}


trainX = data['trainX'].values
trainY = data['trainY'].values
test_X = data['test_X'].values
upup = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

for feature in upup:
	for times in np.arange(2, 20, 1):
		trainX = np.concatenate([trainX, (data['trainX'][feature].values.reshape(-1, 1)**times)], axis=1)
		test_X = np.concatenate([test_X, (data['test_X'][feature].values.reshape(-1, 1)**times)], axis=1)

def normalize(x, y):
	mean = np.mean(x, axis=0).astype(np.float32)
	dev = np.std(x, axis=0).astype(np.float32)
	x = (x-mean) / (dev+1e-10)
	y = (y-mean) / (dev+1e-10)
	return x, y

trainX, test_X = normalize(trainX, test_X)

model = LogisticRegression(penalty='l2', C=1, random_state=7122, max_iter=500)
model.fit(trainX, trainY)
y_pred = model.predict(test_X).reshape(-1, 1)
label = np.arange(y_pred.shape[0]).reshape(-1, 1)
label = label + 1
ans = np.concatenate([label, y_pred], axis=1).astype(int)
np.savetxt(sys.argv[4], ans, fmt="%s", header='id,label', comments='', delimiter=",")

