import numpy as np
import pandas as pd
import argparse
from keras.models import load_model

parser = argparse.ArgumentParser(description='ML HW6')
parser.add_argument('--test', type=str, help='testing data')
parser.add_argument('--model', type=str, help='model file')
args = parser.parse_args()


model = load_model(args.model)

test = pd.read_csv(args.test)


y_pred = model.predict([test['UserID'], test['MovieID']], batch_size=32).reshape(-1, 1)
y_pred = np.clip(y_pred, 1.0, 5.0)
label = np.arange(y_pred.shape[0]).reshape(-1, 1).astype(int)
print ("TestDataID,Rating")
for x, y in zip(label, y_pred):
	print (x[0]+1, end='')
	print (",", end='')
	print (y[0])

