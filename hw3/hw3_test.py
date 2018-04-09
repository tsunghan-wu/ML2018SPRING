import argparse
import numpy as np
import pandas as pd 
from keras.models import load_model

parser = argparse.ArgumentParser(description='ML HW3')
parser.add_argument('test', type=str, help='testing data')
parser.add_argument('out', type=str, help='prediction file')
parser.add_argument('mode', type=str, help='private or public')
args = parser.parse_args()



def read_data(testing):
	test = pd.read_csv(testing)
	return np.array(test['feature'].str.split(" ").values.tolist()).reshape(-1, 48, 48, 1).astype(np.float32)
def normalize(x):
	mean = np.load("mean.npy")
	dev = np.load("dev.npy")
	x = (x-mean) / (dev+1e-10)
	return x

### load testing data ###
testX = read_data(args.test)
testX = normalize(testX)

### load model ###
if args.model == 'public':
	model = load_model("./CNN_Aug.h5")
print (model.summary())

### predict ###
y_pred = model.predict(testX)
y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1)
label = np.arange(y_pred.shape[0]).reshape(-1, 1)
ans = np.concatenate([label, y_pred], axis=1).astype(int)
np.savetxt(args.out, ans, fmt="%s", header='id,label', comments='', delimiter=",")


