import sys
import os
import json
import numpy as np
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from nn_utils import defineLSTM, getDim

from random import randint

def selectWeights(path):
	# load file names
	files = [path + f for f in os.listdir(path) if ".hdf5" in f]

	# parse loss value from file name
	loss_dic = dict((float(re.search("[0-9]\.[0-9]{2,4}", f).group(0)), f) for f in files)

	# return filename with min key value
	min_key = min(loss_dic, key=loss_dic.get)
	return loss_dic.get(min_key)

def seedFile(path, train_type):
	# load file names
	files = [path + f for f in os.listdir(path) if ".json" in f and train_type in f]

	# randomly select file we'll draw seed from
	rand_val = randint(0,len(files)-1)
	return files[rand_val]

def main():

	# set paths
	TRAIN_PATH = "data/training/"
	WEIGHT_PATH = "data/weights/"

	# set arg
	train_type = sys.argv[1]

	files = [TRAIN_PATH + f for f in os.listdir(TRAIN_PATH) if ".json" in f and train_type in f]

	# get training dimensions to feed LSTM
	X_1, X_2, y_1 = getDim(files[0])

	# define model
	model = defineLSTM(X_1, X_2, y_1)

	# select weights
	weight_file = selectWeights(WEIGHT_PATH)

	# load weights
	model.load_weights(weight_file)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	# get seed file
	seed_file = seedFile(TRAIN_PATH, train_type)

	# load file
	with open(seed_file) as f:
		data = json.load(f)

	# get X data, int_to_element
	X = data.get('X') 
	int_to_element = data.get('int_to_element')
	uniq_count = data.get('unique_count')

	# set random seed
	start = np.random.randint(0, len(X)-1)
	pattern = X[start]

	# set seed pattern
	seed_pattern = "".join([int_to_element[str(value)] for value in pattern])
	print("Seed: %s" % seed_pattern)

	text = seed_pattern

	# write text
	for i in range(100):
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(uniq_count)
		prediction = model.predict(x, verbose=0)
		index = np.argmax(prediction)
		result = int_to_element[str(index)]
		text += result
		pattern.append(index)
		pattern=pattern[1:len(pattern)]

	print(text)

if __name__ in "__main__":
	main()




