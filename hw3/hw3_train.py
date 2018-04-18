import sys
import math
import random
import argparse
import numpy as np
import pandas as pd 
def read_data(training):
	train = pd.read_csv(training)
	ret = {
		'trainX': np.array(train['feature'].str.split(" ").values.tolist()).reshape(-1, 48, 48, 1).astype(np.float32),
		'trainY': pd.get_dummies(train['label']).values.astype(int),
	}
	return ret

parser = argparse.ArgumentParser(description='ML HW3')    
parser.add_argument('train', type=str, help='training data')
parser.add_argument('--model', type=str, help='save model path')
parser.add_argument('-s','--scale', type=bool, help='scale data', default=True)
args = parser.parse_args()

data = read_data(args.train)	# dictionary (trainX, trainY) 

print ("finish parsing and read data")


def normalize(x):
	mean = np.mean(x, axis=0).astype(np.float32)
	dev = np.std(x, axis=0).astype(np.float32)
	np.save("./mean.npy", mean)
	np.save("./dev.npy", dev)
	x = (x-mean) / (dev+1e-10)
	return x
if args.scale is True:
	print ("normalize")
	data['trainX'] = normalize(data['trainX'])

#### Training Start ####
from kr_model import model1, model2, model3, model4
from keras.layers import Input, Dense
from keras.models import Model
from keras import layers


M1 = model1(lr=1e-3, decay=5e-6, epoch=500)
M2 = model2(lr=1e-3, decay=5e-6, epoch=200)
M3 = model3(lr=1e-3, decay=5e-6, epoch=200)
M4 = model4(lr=1e-3, decay=5e-6, epoch=200)
M5 = model1(lr=1e-3, decay=5e-6, epoch=900)

M1.train(data['trainX'], data['trainY'])
M2.train(data['trainX'], data['trainY'])
M3.train(data['trainX'], data['trainY'])
M4.train(data['trainX'], data['trainY'])
M5.train(data['trainX'], data['trainY'])


def ensembleModels(ymodel):
	model_input = Input(shape=ymodel[0].input_shape[1:])
	yModels=[model(model_input) for model in ymodel] 
	yAvg=layers.average(yModels)
	modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
	print (modelEns.summary())
	modelEns.save(args.model)

ensembleModels(ymodel=[M1.model, M2.model, M3.model, M4.model, M5.model])
