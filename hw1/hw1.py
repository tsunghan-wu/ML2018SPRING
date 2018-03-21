import sys
import numpy as np
import pandas as pd 

test_path = sys.argv[1]
output_path = sys.argv[2]

test = pd.read_csv(test_path, names=['id', 'che', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
p1 = test.loc[test['che'] == 'PM2.5'].drop(['id', 'che'], axis=1).values.astype(np.float32)
p2 = test.loc[test['che'] == 'PM10'].drop(['id', 'che'], axis=1).values.astype(np.float32)
w = np.load("./hw1.npy")

def add_bias(data):
	row = data.shape[0]
	bias = np.ones(shape=(row, 1), dtype=np.float32)
	return np.concatenate((data, bias), axis=1)

hour = 9 
feature = hour * 2
test = np.concatenate((p2, p1), axis=1)
print (test.shape)
X_test = np.empty(shape=[0, feature])

def invalid(x):
	if x > 120 or x <= 1:
		return True
	return False

for row in test:
	if invalid(row[0]):
		row[0] = (row[1] + row[2])/2
	for i in range(1, row.shape[0]-1):
		if invalid(row[i]):
			row[i] = (row[i-1] + row[i+1])/2
	if invalid(row[-1]):
		row[-1] = (row[-2] + row[-3])/2
	X_test = np.append(X_test, row).reshape(-1, feature)



X_test = add_bias(X_test)
label = np.array(['id_'+str(i) for i in range(260)]).reshape(-1, 1)
y_test = np.dot(X_test, w)
final = np.concatenate((label, y_test), axis=1)
np.savetxt(output_path, final, fmt="%s", header='id,value', comments='', delimiter=",")
