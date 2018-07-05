import sys, csv, os
import pandas as pd
import numpy as np
import keras
# from sklearn.preprocessing import StandardScaler
from keras.utils import *
from keras.layers import Input, Dense, Flatten, Dropout, merge, Embedding, BatchNormalization, Add, Convolution2D, MaxPool2D, Activation, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import non_neg
from keras.metrics import top_k_categorical_accuracy
# from keras.initializers import glorot_normal

print("Keras's version ----> ", keras.__version__)
EPOCH = 200
LOAD_STATUS = False


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def mean_pred(y_true, y_pred):	# Not used
	return K.mean(y_pred)
def audio_norm(data):			# Not used
	max_data = np.max(data)
	min_data = np.min(data)
	data = (data-min_data)/(max_data-min_data+1e-6)
	return data-0.5



def get_model_CNN(nclass, dim1, dim2):
	
	inp = Input(shape=(1, dim1, dim2))
	x = Convolution2D(32, (2,5), data_format='channels_first', padding="same")(inp)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)
	
	x = Convolution2D(64, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)
	
	x = Convolution2D(128, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)
	
	x = Convolution2D(256, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)

	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(128, activation="relu")(x)
	# x = BatchNormalization()(x)

	out = Dense(nclass, activation="softmax")(x)
	model = Model(inputs=inp, outputs=out)

	return model

def get_model_CNN_for13(nclass, dim1, dim2):
	
	inp = Input(shape=(1, dim1, dim2))
	x = Convolution2D(32, (2,5), data_format='channels_first', padding="same")(inp)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)
	
	x = Convolution2D(64, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)
	
	x = Convolution2D(128, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)
	
	x = Convolution2D(256, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)

	x = Convolution2D(384, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPool2D(data_format='channels_first')(x)

	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(128, activation="relu")(x)
	# x = BatchNormalization()(x)

	out = Dense(nclass, activation="softmax")(x)
	model = Model(inputs=inp, outputs=out)

	return model

def get_model_REST(nclass, dim1, dim2):

	def Residual(In, height):
		icut = In
		In = Convolution2D(height//4, kernel_size=1, strides=1, padding='same', data_format='channels_first')(In)
		In = Convolution2D(height//4, kernel_size=(2,5), strides=1, padding='same', data_format='channels_first')(In)
		In = Convolution2D(height, kernel_size=1, strides=1, padding='same', data_format='channels_first')(In)
		In = Add()([In, icut])
		In = Activation('relu')(In)
		return In

	inp = Input(shape=(1, dim1, dim2))
	x = Convolution2D(32, (2,5), data_format='channels_first', padding="same")(inp)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)	
	x = MaxPool2D(data_format='channels_first')(x)
	x = Dropout(0.5)(x)

	x = Convolution2D(64, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	for _ in range(2):
		x = Residual(x, 64)

	x = Convolution2D(128, (2,5), data_format='channels_first', padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	for _ in range(2):
		x = Residual(x, 128)
	x = AveragePooling2D(data_format='channels_first')(x)


	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(128, activation="relu")(x)
	# x = BatchNormalization()(x)

	out = Dense(nclass, activation="softmax")(x)
	model = Model(inputs=inp, outputs=out)

	return model

def get_model_CNN_NoPooling(nclass, dim1, dim2):
	
	inp = Input(shape=(1, dim1, dim2))
	x = Convolution2D(32, (2,5), data_format='channels_first', padding="valid")(inp)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	# x = AveragePooling2D(data_format='channels_first')(x)
	# x = Dropout(0.2)(x)

	x = Convolution2D(64, (2,5), data_format='channels_first', padding="valid")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	# x = AveragePooling2D(data_format='channels_first')(x)
	# x = Dropout(0.2)(x)

	x = Convolution2D(128, (2,5), data_format='channels_first', padding="valid")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = AveragePooling2D(data_format='channels_first')(x)
	# x = Dropout(0.2)(x)

	x = Convolution2D(256, (2,5), data_format='channels_first', padding="valid")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = AveragePooling2D(data_format='channels_first')(x)
	# x = Dropout(0.2)(x)

	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(128, activation="relu")(x)
	x = BatchNormalization()(x)

	out = Dense(nclass, activation="softmax")(x)
	model = Model(inputs=inp, outputs=out)

	return model

## Train Data Loading
Y = np.load(sys.argv[2])
X = np.expand_dims(np.load(sys.argv[1]), axis=1)
print(X.shape)
Y = np_utils.to_categorical(Y, num_classes=41)

X_train, X_value, Y_train, Y_value \
= train_test_split(X, Y, test_size = 0.1, random_state = 17)

# create the model
if LOAD_STATUS:
	print("---- Loading model... ----")
	with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
		model = load_model(sys.argv[3])
else:
	# model = get_model_CNN(41, X.shape[2], X.shape[3])
	model = get_model_CNN_for13(41, X.shape[2], X.shape[3])
	# model = get_model_REST(41, X.shape[2], X.shape[3])
	# model = get_model_CNN_NoPooling(41, X.shape[2], X.shape[3])

	adam = Adam(1e-4)
	model.compile(optimizer=adam,
		loss=['categorical_crossentropy'],
		metrics=['acc', top_3_accuracy],
		loss_weights=None)

filepath = sys.argv[3]
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, early_stopping]
# print(model.summary())
history = model.fit(X_train, Y_train,
					validation_data=(X_value, Y_value),
					epochs=EPOCH,
					batch_size=32,
					verbose=1,
					shuffle=True,
					callbacks=callbacks_list
				)

# model.save(sys.argv[3])
