import math
import numpy as np
import pandas as pd

data = {
	'train': pd.read_csv('./input/train.csv', encoding='big5').drop(['測站'], axis=1).rename(columns={'測項':'che', '日期':'day'}),
	'test': pd.read_csv('./input/test.csv', names=['id', 'che', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
}

##### get feature #####

tr = [data['train'].loc[data['train']['che'] == s].drop(['day', 'che'], axis=1).values.astype(np.float32).reshape(-1, 480)  for s in ["PM2.5", "PM10"]]
te = [data['test'].loc[data['test']['che'] == s].drop(['id', 'che'], axis=1).values.astype(np.float32) for s in ["PM2.5", "PM10"]]
##### split hour #####
Row = tr[0].shape[1]
hour = 9
feature = hour * 2
Train = np.empty(shape=[0, feature+1])

for col in range(0, Row-hour-1):
	last = np.concatenate((tr[1][:,col:col+hour], tr[0][:,col:col+hour+1]), axis=1)
	Train = np.append(Train, last).reshape(-1, feature+1)



X_Train = np.empty(shape=[0, feature])
Y_Train = np.empty(shape=[0, 1])


def valid(row):
	x = row[:-1]
	y = row[-1]
	if len(row[row>120]) > 0 or len(row[row<=1]) > 0:
		return False
	if y > np.max(x) + 3 or y < np.min(x) - 3:
		return False
	return True

for row in Train:
	if (valid(row)):
		row = row.reshape(1, row.shape[0])
		X_Train = np.append(X_Train, row[:,:-1]).reshape(-1, feature)
		Y_Train = np.append(Y_Train, row[:,-1]).reshape(-1, 1)


def add_bias(data):
	row = data.shape[0]
	bias = np.ones(shape=(row, 1), dtype=np.float32)
	return np.concatenate((data, bias), axis=1)

def calculate(X, Y, w):
	Y_pred = np.dot(X, w)
	ans = (np.dot(np.dot(X.T, X), w) - np.dot(X.T, Y)) / X.shape[0]
	return ans

def RMSE(X, Y, w):
	Y_pred = np.dot(X, w)
	Sum = np.sum(np.power((Y_pred-Y), 2))
	return math.sqrt(Sum/Y_pred.shape[0])	

def GD(X, Y, epoch, lr):
	w = np.zeros(shape=(X.shape[1], 1), dtype=np.float32)
	for _ in range(epoch):
		gradient = calculate(X, Y, w)
		w = w - lr * gradient
		error = RMSE(X, Y, w)
		if _ % 100 == 0:
			print (error)
	return w

lr = 0.000001
epoch = 2000000

X_Train = add_bias(X_Train)
w = GD(X_Train, Y_Train, epoch, lr)
print (w)
print (RMSE(X_Train, Y_Train, w))
np.save("./hw1.npy", w)

