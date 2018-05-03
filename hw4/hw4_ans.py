import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='ML HW4')    
parser.add_argument('test', type=str, help='test case')
parser.add_argument('out', type=str, help='prediction file')
args = parser.parse_args()

test = pd.read_csv(args.test).values

label = np.load("./label.npy")

y_pred = []
for x in test:
	l1 = label[x[1]]
	l2 = label[x[2]]
	if (l1 == l2):
		y_pred.append(1)
	else:
		y_pred.append(0)

y_pred = np.array(y_pred).reshape(-1, 1)
label = np.arange(y_pred.shape[0]).reshape(-1, 1)
ans = np.concatenate([label, y_pred], axis=1).astype(int)
np.savetxt(args.out, ans, fmt="%s", header="ID,Ans", comments="", delimiter=",")




