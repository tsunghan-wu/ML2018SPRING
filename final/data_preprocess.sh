#!/bin/bash
mkdir data # the data after preprocess
python3 ./src/preprocessing.py 1 
python3 ./src/preprocessing.py 2
python3 ./src/preprocessing.py 3
python3 ./src/preprocessing.py 5
python3 ./src/preprocessing.py 8
python3 ./src/preprocessing.py 13