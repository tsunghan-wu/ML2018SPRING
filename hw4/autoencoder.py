import numpy as np
import tensorflow as tf
def init_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x, W):
	return tf.nn.conv2d(
		x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(
    	x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(
		x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')


def model(x):
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([5,5,1,16])
	b_conv1 = init_weights([16])
	h_conv1 = tf.nn.selu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([3,3,16,16])
	b_conv2 = init_weights([16])
	h_conv2 = tf.nn.selu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	#### Flatten layer ####
	h_flat = tf.reshape(h_pool2, [ -1 , 7 * 7 * 16])
	w_fc0 = init_weights([ 7 * 7 * 16 , 50])
	b_fc0 = init_weights([50])	
	#### encode layer ####	
	encode = tf.matmul(h_flat, w_fc0) + b_fc0
	tmp = tf.nn.selu(encode)
	#### dense1 ####
	w_fc1 = init_weights([50 , 200])
	b_fc1 = init_weights([200])
	h_fc1 = tf.nn.selu(tf.matmul(tmp , w_fc1) + b_fc1)

	#### output ####
	W_fc2 = init_weights([200, 784])
	b_fc2 = init_weights([784])
	y_conv = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

	return y_conv, encode
