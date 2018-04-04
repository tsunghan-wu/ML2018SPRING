import sys
import math
import pickle
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
'''
[hw2 best method]
- power some column
- feature scaling (standardiza)
- sklearn logistic regression
- regularization
'''
data = {
	'trainX': pd.read_csv(sys.argv[1]),
	'trainY': pd.read_csv(sys.argv[2], names=['y']),
}


trainX = data['trainX'].values
trainY = data['trainY'].values
upup = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

for feature in upup:
	for times in np.arange(2, 100, 1):
		trainX = np.concatenate([trainX, (data['trainX'][feature].values.reshape(-1, 1)**times)], axis=1)
### normalize ###
def normalize(x):
	mean = np.mean(x, axis=0).astype(np.float32)
	dev = np.std(x, axis=0).astype(np.float32)
	np.save("./best_mean.npy", mean)
	np.save("./best_dev.npy", dev)
	x = (x-mean) / (dev)
	return x 

trainX = normalize(trainX)

model = LogisticRegression(penalty='l1', C=3.9, random_state=7122, max_iter=300)
model.fit(trainX, trainY)
with open('./model.pickle', 'wb') as f:
	pickle.dump(model, f)

