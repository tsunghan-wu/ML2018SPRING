from skimage import io
import numpy as np
import argparse
import os 

parser = argparse.ArgumentParser(description='ML HW4')    
parser.add_argument('path', type=str, help='image path')
parser.add_argument('query', type=str, help='query image')
args = parser.parse_args()

img_path = args.path
target = args.query

def img_processing(x):
	x -= np.min(x)
	x /= np.max(x)
	x = (x * 255).astype(np.uint8)
	return x

def read_data(img_path, target):
	# read data
	whole_file = [os.path.join(img_path, file) for file in os.listdir(img_path)]
	target_path = os.path.join(img_path, target)
	whole_img = np.array([io.imread(x) for x in whole_file]).astype(np.float64)
	target_img = np.array(io.imread(target_path)).astype(np.float64)
	# average
	avg = np.mean(whole_img, axis=0).astype(np.float64)
	whole_img -= avg
	target_img -= avg
	return whole_img, target_img, avg


whole_img, target_img, avg = read_data(img_path, target)

target_img = target_img.reshape(-1)

# svd
eigface, eigval, _ = np.linalg.svd(whole_img.reshape(-1, 600*600*3).T, full_matrices=False)

num_component = 4

# reconstruct
eigf4 = eigface[:,:num_component]
weight = np.dot(target_img, eigf4)

reconstruct = np.zeros(600*600*3)
for i in range(num_component):
	reconstruct += eigf4[:,i] * weight[i]

reconstruct = reconstruct.reshape(600, 600, 3)
reconstruct += avg
io.imsave('reconstruct.jpg', img_processing(reconstruct))