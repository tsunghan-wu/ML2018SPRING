import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import autoencoder
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='ML HW4')    
parser.add_argument('image', type=str, help='image npy file')
parser.add_argument('test', type=str, help='test case')
parser.add_argument('out', type=str, help='prediction file')
args = parser.parse_args()

image = np.load(args.image).astype(np.float64)
test = pd.read_csv(args.test).values
image /= 255
x = tf.placeholder(tf.float32, [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 28*28])

y, encoder = autoencoder.model(x)

#### model compile ####
bce = tf.keras.backend.binary_crossentropy(target=y_,output=y)
train_step = tf.train.AdamOptimizer(0.001).minimize(bce)

def next_batch(batch_size):
	tmp = np.random.permutation(image.shape[0])[:batch_size]
	X = image[tmp]
	return X, X

with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	saver = tf.train.Saver()
	saver.restore(sess, "./model/auto.ckpt")


	size = image.shape[0]
	feature = []
	for i in range(size):
		ret = sess.run(encoder, feed_dict={
				x: image[i].reshape(-1, 28*28)
			})

		feature.append(ret)

	feature = np.array(feature).reshape(size, -1)

km = KMeans(n_clusters=2, random_state=7122)
label = km.fit_predict(feature)
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




