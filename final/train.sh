#!/bin/bash

python3 ./src/train.py "./data/trainX"$1".npy" "./data/trainY"$1".npy" $2
