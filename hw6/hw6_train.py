import numpy as np
import pandas as pd
from keras import regularizers
from keras.models import Model, Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Flatten, Reshape, Dot, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import non_neg

input_file = "../input/train.csv"
data = pd.read_csv(input_file)

def __graph__(uc, mc, es):
    user_input = Input(shape=[1])
    user_emb = Flatten()(Embedding(uc, es, embeddings_initializer='glorot_normal',)(user_input))
    user_emb = Dropout(0.2)(user_emb)
    movie_input = Input(shape=[1])
    movie_emb = Flatten()(Embedding(mc, es, embeddings_initializer='glorot_normal',)(movie_input))
    movie_emb = Dropout(0.2)(movie_emb)

    pred = Dot(axes=1)([movie_emb, user_emb])

    return Model([user_input, movie_input], pred)


user_count = max(data['UserID']) + 1
movie_count = max(data['MovieID']) + 1
model = __graph__(user_count, movie_count, es=50)

adam = Adam(lr=5e-4, decay=5e-6)
model.compile(adam, 'mse')
print (model.summary())

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

save_path = "model_50.h5"

earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(
	filepath=save_path, verbose=1, save_best_only=True,monitor='val_loss', mode='auto')

np.random.seed(7122)
data = data.values[:,1:]
np.random.shuffle(data)



train_data, test_data = train_test_split(data, test_size=0.05, random_state=7122)


model.fit([train_data[:,0], train_data[:,1]], train_data[:,2], epochs=100, batch_size=64, 
	shuffle=True, validation_data=([test_data[:,0], test_data[:,1]], test_data[:,2])
	,callbacks=[checkpoint, earlystopping])



