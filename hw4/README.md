## ML HW4

### Part 1 PCA of colored faces

1. execute: `bash pca.sh <image directory> <query file>`
2. result: get one image named `reconstruct.jpg` on the same folder

### Part 2 Image clustering

1. execute: `bash hw4.sh <image.npy> <test case> <prediction file>`
2. result: prediction file (which you can submit to kaggle)
3. `autoencoder.py` is a file needed to be import in both `hw4_train.py, hw4_test.py`
4. model is a directory which stores my model (checkpoint)

### Part 3 Ensemble learning

1. `ensemble.py, kr_model.py` are training file (ensemble method is included)
2. `dev.npy, mean.npy` are files you need if you want to execute `ensemble_test.py` (you can not execute it directly because there is no model)
