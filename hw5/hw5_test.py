import pandas as pd
import argparse
import numpy as np 
import itertools
import pickle
from keras.models import load_model

parser = argparse.ArgumentParser(description='ML HW5')
parser.add_argument('test', type=str, help='testing data')
parser.add_argument('out', type=str, help='prediction file')
parser.add_argument('mode', type=str, help='private or public')
args = parser.parse_args()


# testing_file = "../input/testing_data.txt"
testing_file = args.test
model = load_model('x_final_ensemble.h5')
slen = 30
with open("./word2vec_model.pickle", "rb") as f:
	gensim_model = pickle.load(f)

with open(testing_file, 'r') as f:
	testing_data = f.read()
testing_data = testing_data.split('\n')
whole_data = []
for s in testing_data[1:-1]:
	s = s.split(',')[1:]
	s = ''.join(s)
	s = s.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt").replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont").replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt").replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt").replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt").replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ").replace("couldn ' t","couldnt")
	s = ''.join(i for i, _ in itertools.groupby(s))
	news = []
	s = s.split(' ')

	whole_data.append(s)


def word2vec(rawX):
	X = []
	for line in rawX:
		news = []
		for word in line:
			try: 
				vec = gensim_model[word]
				news.append(vec)
			except:
				pass
		news = news[:slen]
		news += [np.zeros([200,]) for _ in range(slen-len(news))]
		X.append(news)
	return X

batch_counter = 0
batch_size = 50
def generator():
	global batch_counter
	while True:
		if batch_counter + batch_size >= len(whole_data):
			X = word2vec(whole_data[batch_counter:])
			batch_counter = 0
		else:
			X = word2vec(whole_data[batch_counter:batch_counter+batch_size])
			batch_counter += batch_size
		print (batch_counter)
		yield np.array(X)

y_pred = model.predict_generator(generator(), steps=len(whole_data)//batch_size)

y_pred = np.array(y_pred).reshape(-1, 1)
y_pred[y_pred>=0.5] = 1
y_pred[y_pred < 0.5] = 0
label = np.arange(y_pred.shape[0]).reshape(-1, 1)
ans = np.concatenate([label, y_pred], axis=1).astype(int)
np.savetxt(args.out, ans, fmt="%s", header='id,label', comments='', delimiter=",")


