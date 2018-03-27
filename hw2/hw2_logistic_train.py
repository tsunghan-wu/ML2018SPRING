import sys
import math
import numpy as np 
import pandas as pd

data = {
	'trainX': np.loadtxt(sys.argv[1], skiprows=1, delimiter=','),
	'trainY': np.loadtxt(sys.argv[2]).reshape(-1, 1),
}

### Normalization ###
def normalize(x):
	mean = np.mean(x, axis=0).astype(np.float32)
	dev = np.std(x, axis=0)
	np.save("./mean.npy", mean)
	np.save("./dev.npy", dev)
	x = (x-mean) / dev
	return x 

data['trainX'] = normalize(data['trainX'])

### Extract data ### 
useless = [10]
data['trainX'] = np.delete(data['trainX'], useless, axis=0)

age = [np.power(data['trainX'][:,0], s).reshape(-1, 1) for s in [2, 3]]
female = [np.power(data['trainX'][:,75], s).reshape(-1, 1) for s in [2, 3]]
male = [np.power(data['trainX'][:,76], s).reshape(-1, 1) for s in [2, 3]]
data['trainX'] = np.concatenate([data['trainX']] + age + female + male, axis=1)

### LOGISTIC REGRESSION ###

# hyper parameter #
lr = 6
Lambda = 2e-6

def add_bias(data):
	row = data.shape[0]
	bias = np.ones(shape=(row, 1), dtype=np.float32)
	return np.concatenate((bias, data), axis=1)

def sigmoid_function(c):
	return 1.0 / (1+np.exp(-c))

def log_reg(x, y, yeeeeta, times):
	row, col = x.shape
	W = np.zeros((col, 1), dtype=np.float)
	s_gra = np.zeros((col, 1), dtype=np.float)
	for t in range(times):
		sigmoid = sigmoid_function(-1*y*x.dot(W))
		gradient_E = -(y*x).T.dot(sigmoid)/float(row)
		s_gra += gradient_E**2
		ada = np.sqrt(s_gra)
#		acc = Acc(x, y, W)
#		if t % 50 == 0:
#			print ("epoch = ", t, "acc = ", acc, "norm = ", np.linalg.norm(W))
		W = W - (yeeeeta/ada)*(gradient_E + Lambda * W)
	return W

def Acc(testdata, y, ww):
	row, col = testdata.shape
	y_predict = np.sign(np.dot(testdata, ww))
	return np.sum(y_predict == y) / float(row)

data['trainY'][data['trainY']==0] = -1
data['trainX'] = add_bias(data['trainX'])
w = log_reg(data['trainX'], data['trainY'], lr, 5000)
np.save("./logistic_regression.npy", w)
