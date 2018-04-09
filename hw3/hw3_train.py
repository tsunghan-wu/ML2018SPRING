import sys
import math
import random
import argparse
import numpy as np
import pandas as pd 

def read_data(training, testing):
	train = pd.read_csv(training)
	test = pd.read_csv(testing)
	ret = {
		'trainX': np.array(train['feature'].str.split(" ").values.tolist()).reshape(-1, 48*48).astype(np.float32),
		'trainY': pd.get_dummies(train['label']).values.astype(int),
	}
	return ret

parser = argparse.ArgumentParser(description='ML HW3')    
parser.add_argument('train', type=str, help='training data')
parser.add_argument('model', type=str, help='save model path')
parser.add_argument('-s','--scale', type=bool, help='scale data', default=True)
args = parser.parse_args()

data = read_data(args.train)	# dictionary (trainX, trainY) 

print ("finish parsing and read data")


def normalize(x):
	mean = np.mean(x, axis=0).astype(np.float32)
	dev = np.std(x, axis=0).astype(np.float32)
	x = (x-mean) / (dev+1e-10)
	return x
if args.scale is True:
	print ("normalize")
	data['trainX'] = normalize(data['trainX'])

data['trainX'] = data['trainX'].reshape(-1, 48, 48, 1)
data['test_X'] = data['test_X'].reshape(-1, 48, 48, 1)

#### Training Start ####

from kr_model import CNN
model = CNN(lr=1e-3, decay=5e-6, epoch=300, save=args.model)
model.train(data['trainX'], data['trainY'])
model.save_model()


