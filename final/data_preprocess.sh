#!/bin/bash
mkdir data # the data after preprocess
python3 ./src/preprocessing.py 1 $1
python3 ./src/preprocessing.py 2 $1
python3 ./src/preprocessing.py 3 $1
python3 ./src/preprocessing.py 5 $1
python3 ./src/preprocessing.py 8 $1
python3 ./src/preprocessing.py 13 $1
