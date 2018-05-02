import numpy as np
import pandas as pd
import tensorflow as tf
import autoencoder
from sklearn.cluster import KMeans


image = np.load("../input/image.npy").astype(np.float64)
test = pd.read_csv("../input/test_case.csv").values
image /= 255
x = tf.placeholder(tf.float32, [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 28*28])

y, encoder = autoencoder.model(x)

#### model compile ####
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
# mse = tf.reduce_mean(tf.square( y_ - y ))
bce = tf.keras.backend.binary_crossentropy(target=y_,output=y)
train_step = tf.train.AdamOptimizer(0.001).minimize(bce)

def next_batch(batch_size):
	tmp = np.random.permutation(image.shape[0])[:batch_size]
	X = image[tmp]
	return X, X

o = open("./log.txt", 'w')

with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)

	for i in range(30000):
		batch = next_batch(128)
		sess.run(train_step , feed_dict={
				x : batch[0],
				y_: batch[1],
			})
		if i % 200 == 199:
			loss = sess.run(bce, feed_dict={
					x : batch[0],
					y_: batch[1],
				})
			print ("iteration = ", i, "loss = ", loss, file=o)
			o.flush()
	saver = tf.train.Saver()
	saver.save(sess,"./model/auto.ckpt")

	size = image.shape[0]
	feature = []
	for i in range(size):
		ret = sess.run(encoder, feed_dict={
				x: image[i].reshape(-1, 28*28)
			})

		feature.append(ret)

	feature = np.array(feature).reshape(size, -1)

# km = KMeans(n_clusters=2)
# label = km.fit_predict(feature)
# y_pred = []
# for x in test:
# 	l1 = label[x[1]]
# 	l2 = label[x[2]]
# 	if (l1 == l2):
# 		y_pred.append(1)
# 	else:
# 		y_pred.append(0)


# y_pred = np.array(y_pred).reshape(-1, 1)
# label = np.arange(y_pred.shape[0]).reshape(-1, 1)
# ans = np.concatenate([label, y_pred], axis=1).astype(int)
# np.savetxt("tf.csv", ans, fmt="%s", header="ID,Ans", comments="", delimiter=",")




