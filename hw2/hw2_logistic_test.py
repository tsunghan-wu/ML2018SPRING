import sys
import numpy as np 
import pandas as pd

test = np.loadtxt(sys.argv[1], skiprows=1, delimiter=',')
w = np.load("./logistic_regression.npy")

### normalize ###
def normalize(x):
	mean = np.load("./mean.npy")
	dev = np.load("./dev.npy")
	x = (x-mean) / (dev)
	return x 

test = normalize(test)

### extract feature ###
test = np.delete(test, [10], axis=1)
age = [np.power(test[:,0], s).reshape(-1, 1) for s in [2, 3]]
female = [np.power(test[:,75], s).reshape(-1, 1) for s in [2, 3]]
male = [np.power(test[:,76], s).reshape(-1, 1) for s in [2, 3]]
test = np.concatenate([test] + age + female + male, axis=1)

### predict ###
def add_bias(data):
	row = data.shape[0]
	bias = np.ones(shape=(row, 1), dtype=np.float32)
	return np.concatenate((bias, data), axis=1)

test = add_bias(test)
y_pred = np.sign(np.dot(test, w))
y_pred[y_pred == -1] = 0
label = np.arange(y_pred.shape[0]).reshape(-1, 1)
label = label + 1
ans = np.concatenate([label, y_pred], axis=1).astype(int)
np.savetxt(sys.argv[2], ans, fmt="%s", header='id,label', comments='', delimiter=",")
