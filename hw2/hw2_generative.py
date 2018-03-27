import sys
import math
import numpy as np

data = {
	'trainX': np.loadtxt(sys.argv[1], skiprows=1, delimiter=','),
	'trainY': np.loadtxt(sys.argv[2]).reshape(-1, 1),
	'test_X': np.loadtxt(sys.argv[3], skiprows=1, delimiter=',')
}
### Normalize feature ###
def normalize(x, y):
	mean = np.mean(x, axis=0).astype(np.float32)
	dev = np.std(x, axis=0).astype(np.float32)
	x = (x-mean) / (dev)
	y = (y-mean) / (dev)
	return x, y

data['trainX'], data['test_X'] = normalize(data['trainX'], data['test_X'])

### Extract important feature ###

def extract(x):
	tmp = [x[:,s].reshape(-1, 1) for s in useful]
	return np.concatenate(tmp, axis=1)

useless = [7, 8, 10] + [i for i in range(27, 50)] + [54, 56, 65, 67] + [i for i in range(81, 123)]
idx = [i for i in range(123)]
useful = [x for x in idx if x not in useless]
data['trainX'] = extract(data['trainX'])
data['test_X'] = extract(data['test_X'])


age = [np.power(data['trainX'][:,0], s).reshape(-1, 1) for s in [2, 3]]
data['trainX'] = np.concatenate([data['trainX']] + age , axis=1)

tage = [np.power(data['test_X'][:,0], s).reshape(-1, 1) for s in [2, 3]]
data['test_X'] = np.concatenate([data['test_X']] + tage, axis=1)


### generative model ###
train = np.concatenate([data['trainX'], data['trainY']], axis=1)
ONE = train[train[:,-1]==1]
ZERO = train[train[:,-1]==0]

def cov(train, mean):
	mat = np.zeros([mean.shape[0], mean.shape[0]], dtype=np.float32)
	for row in train[:,:-1]:
		x = row.reshape(-1, 1)
		mat = mat + np.dot((x-mean), (x-mean).T)
	return mat / train.shape[0]

P = {
	'zero': (ZERO.shape[0]/train.shape[0]),
	'one': (ONE.shape[0]/train.shape[0])
}

Mean = {	# mean vector
	'zero': ZERO[:,:-1].mean(axis=0).reshape(-1, 1),
	'one': ONE[:,:-1].mean(axis=0).reshape(-1, 1),
}

Mat = {
	'zero': cov(ZERO, Mean['zero']),
	'one': cov(ONE, Mean['one'])
}

Cov_mat = Mat['one'] * P['one'] + Mat['zero'] * P['zero']	# Cov matrix
Cov_rev = np.linalg.pinv(Cov_mat)

def Zero(x):
	return np.exp(-0.5 * np.dot(np.dot((x-Mean['zero']).T, Cov_rev), (x-Mean['zero'])))

def One(x):
	return np.exp(-0.5 * np.dot(np.dot((x-Mean['one']).T, Cov_rev), (x-Mean['one'])))

def predict(X):
	y_pred = []
	for row in X:
		x = row.reshape(-1, 1)
		if (Zero(x) * P['zero'] > 1 * One(x) * P['one']):
			y_pred.append(0)
		else:
			y_pred.append(1)
	return np.array(y_pred).reshape(-1, 1)


### predict ###
y_pred = predict(data['test_X'])
label = np.arange(y_pred.shape[0]).reshape(-1, 1)
label = label + 1
ans = np.concatenate([label, y_pred], axis=1).astype(int)
np.savetxt(sys.argv[4], ans, fmt="%s", header='id,label', comments='', delimiter=",")

