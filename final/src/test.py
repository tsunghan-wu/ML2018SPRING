import sys, csv, os
import pandas as pd
import numpy as np
import keras
from keras.models import Model, load_model
from keras.metrics import top_k_categorical_accuracy
from keras.utils import CustomObjectScope
from keras import backend as K
root_dir = sys.argv[1]
output_path = sys.argv[2]
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
label_map = ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock', 
			'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard', 'Keys_jangling', 
			'Snare_drum', 'Writing', 'Laughter', 'Tearing', 'Fart', 'Oboe', 'Flute', 
			'Cough', 'Telephone', 'Bark', 'Chime', 'Bass_drum', 'Bus', 'Squeak', 'Scissors', 
			'Harmonica', 'Gong', 'Microwave_oven', 'Burping_or_eructation', 'Double_bass', 
			'Shatter', 'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano', 'Meow', 
			'Drawer_open_or_close', 'Applause', 'Acoustic_guitar', 'Violin_or_fiddle', 'Finger_snapping']

def top_3_accuracy(y_true, y_pred):
	return top_k_categorical_accuracy(y_true, y_pred, k=3)
Output=[]




X_test = np.expand_dims(np.load("./data/testX13.npy"), axis=1)
print(X_test.shape)
with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
	for i in [1, 2, 3, 5, 6, 7, 8, 9]:
		model = load_model("./needed_model/cnn_13_v%d.h5" %(i))
		# print(model.summary())
		k = model.predict(X_test, verbose=1)
		print(k[0])
		Output.append(k)
		K.clear_session()

X_test = np.expand_dims(np.load("./data/testX1.npy"), axis=1)
print(X_test.shape)
with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
	for i in range(1):
		model = load_model("./needed_model/cnn_1_v%d.h5" %(i+1))
		# print(model.summary())
		k = model.predict(X_test, verbose=1)
		print(k[0])
		Output.append(k)
		K.clear_session()

X_test = np.expand_dims(np.load("./data/testX2.npy"), axis=1)
print(X_test.shape)
with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
	for i in [0, 2]:
		model = load_model("./needed_model/CNN_40_2_ver%d.h5" %(i+1))
		# print(model.summary())
		k = model.predict(X_test, verbose=1)
		print(k[0])
		Output.append(k)
		K.clear_session()

with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
	model = load_model("./needed_model/cnn_relu_dropout_2.h5")
	# print(model.summary())
	k = model.predict(X_test, verbose=1)
	print(k[0])
	Output.append(k)
	K.clear_session()


X_test = np.expand_dims(np.load("./data/testX3.npy"), axis=1)
print(X_test.shape)
with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
	for i in range(2):
		model = load_model("./needed_model/cnn_3_v%d.h5" %(i+1))
		# print(model.summary())
		k = model.predict(X_test, verbose=1)
		print(k[0])
		Output.append(k)
		K.clear_session()

X_test = np.expand_dims(np.load("./data/testX5.npy"), axis=1)
print(X_test.shape)
with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
	for i in range(1, 9):
		model = load_model("./needed_model/cnn_5_v%d.h5" %(i))
		# print(model.summary())
		k = model.predict(X_test, verbose=1)
		print(k[0])
		Output.append(k)
		K.clear_session()

X_test = np.expand_dims(np.load("./data/testX8.npy"), axis=1)
print(X_test.shape)
with CustomObjectScope({'top_3_accuracy': top_3_accuracy}):
	for i in [1, 2, 3, 5, 6, 7, 8]:
		model = load_model("./needed_model/cnn_8_v%d.h5" %(i))
		# print(model.summary())
		k = model.predict(X_test, verbose=1)
		print(k[0])
		Output.append(k)
		K.clear_session()


y_pred = np.average(Output, axis=0)
def idx2label(y_pred):
    y_pred = np.argsort(y_pred, axis=1)[:,::-1]
    y_pred = y_pred[:,:3]
    ret = []
    for x in y_pred:
        pred = [label_map[y] for y in x]
        pred = " ".join(pred)
        ret.append(pred)
    return ret

tmp = pd.read_csv(os.path.join(root_dir, "sample_submission.csv"))
fname = tmp['fname']
label = pd.DataFrame({'label':idx2label(y_pred)})
fname = pd.DataFrame({'fname':fname})
ans = pd.concat([fname, label], axis=1)
ans.to_csv(output_path, sep=',', index=False)
