import sys
import numpy as np
import pandas as pd 

test_path = sys.argv[1]
output_path = sys.argv[2]

test = pd.read_csv(test_path, names=['id', 'che', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
TEST = test.loc[test['che'] == 'PM2.5'].drop(['id', 'che'], axis=1).values.astype(np.float32)
w = np.load("./linreg_w.npy")


def add_bias(data):
	row = data.shape[0]
	bias = np.ones(shape=(row, 1), dtype=np.float32)
	return np.concatenate((data, bias), axis=1)

hour = 9
X_test = TEST[:,-hour:]
X_test = add_bias(X_test)
label = np.array(['id_'+str(i) for i in range(260)]).reshape(-1, 1)
y_test = np.dot(X_test, w)
final = np.concatenate((label, y_test), axis=1)
np.savetxt(output_path, final, fmt="%s", header='id,value', comments='', delimiter=",")