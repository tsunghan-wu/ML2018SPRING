import pandas as pd
import numpy as np 
import itertools
import pickle
import gensim
import argparse

parser = argparse.ArgumentParser(description='ML HW5')
parser.add_argument('train', type=str, help='testing data')
parser.add_argument('semi', type=str, help='testing data')
args = parser.parse_args()
training_file = args.train
semi_file = args.semi
slen = 30

# load training data + semi data
with open(training_file, 'r') as f:
	train_data = f.read()
train_data = train_data.split('\n')
whole_data = []
trainY = []
for x in train_data[:-1]:
	tmp = x.split(' +++$+++ ')
	s = tmp[1]
	s = s.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt").replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont").replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt").replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt").replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt").replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ").replace("couldn ' t","couldnt")
	trainY.append(x[0])
	s = ''.join(i for i, _ in itertools.groupby(s))
	s = s.split(' ')
	whole_data.append(s)

with open(semi_file, 'r') as f:
	semi_data = f.read()
semi_data = semi_data.split('\n')
for s in semi_data[:-1]:
	s = ''.join(i for i, _ in itertools.groupby(s))
	s = s.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt").replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont").replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt").replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt").replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt").replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ").replace("couldn ' t","couldnt")
	s = s.split(' ')
	whole_data.append(s)

trainx = whole_data[:200000]
semi = whole_data[200000:]

# dump gensim model
gensim_model = gensim.models.Word2Vec(whole_data, size=200, workers = 4, hs=1, negative=0)
dictionary = {}
for x in gensim_model.wv.vocab:
	dictionary.update({x:gensim_model.wv[x]})
with open ("./word2vec_model.pickle", "wb") as f:
	pickle.dump(dictionary, f)

print ("------------finish dictionary model------------")
# start RNN training
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Conv2D, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from keras import layers

def word2vec(rawX):
	X = []
	for line in rawX:
		news = []
		for word in line:
			try: 
				vec = dictionary[word]
				news.append(vec)
			except:
				pass
		news = news[:slen]
		news += [np.zeros([200,]) for _ in range(slen-len(news))]
		X.append(news)
	X = np.array(X)
	return X

def rnn1():
	model = Sequential()
	model.add(LSTM(256,activation="tanh",dropout=0.3,return_sequences = False,
			kernel_initializer='Orthogonal', input_shape=(30, 200)))
	# model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False,
	# 		kernel_initializer='Orthogonal'))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def rnn2():
	model = Sequential()
	model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True,
			kernel_initializer='Orthogonal', input_shape=(30, 200)))
	model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False,
			kernel_initializer='Orthogonal'))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def rnn3():
	model = Sequential()
	model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True,
			kernel_initializer='Orthogonal', input_shape=(30, 200)))
	model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True,
			kernel_initializer='Orthogonal'))
	model.add(Reshape([30, 128, 1]))
	model.add(Conv2D(filters=8, kernel_size=(2, 2), padding='same'))
	model.add(BatchNormalization(axis=-1, momentum=0.5))
	model.add(LeakyReLU())
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def rnn4():
	model = Sequential()
	model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True,
			kernel_initializer='Orthogonal', input_shape=(30, 200)))
	model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False,
			kernel_initializer='Orthogonal')))
	# model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False,
	# 		kernel_initializer='Orthogonal'))

	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

model1 = rnn1()
model2 = rnn2()
model3 = rnn3()
# model4 = rnn4()

batch_counter = 0
batch_size = 50
def generator():
	global batch_counter
	while True:
		if batch_counter + batch_size >= len(trainx):
			X = word2vec(trainx[batch_counter:])
			Y = trainY[batch_counter:]
			batch_counter = 0
		else:
			X = word2vec(trainx[batch_counter:batch_counter+batch_size])
			Y = trainY[batch_counter:batch_counter+batch_size]
			batch_counter += batch_size
		yield (np.array(X), np.array(Y))

model1.fit_generator(generator(), steps_per_epoch=len(trainx)//batch_size, epochs=8)
model2.fit_generator(generator(), steps_per_epoch=len(trainx)//batch_size, epochs=8)
model3.fit_generator(generator(), steps_per_epoch=len(trainx)//batch_size, epochs=8)
# model4.fit_generator(generator(), steps_per_epoch=len(trainx)//batch_size, epochs=8)


def ensembleModels(ymodel):
	model_input = Input(shape=ymodel[0].input_shape[1:])
	yModels=[model(model_input) for model in ymodel] 
	yAvg=layers.average(yModels)
	modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
	print (modelEns.summary())
	modelEns.save("x_final_ensemble.h5")

ensembleModels(ymodel=[model1, model2, model3])

