import sys
import os
import json
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

def normalize(X, y, uniq_count, seq_length):
	"""
	normalize() converts X,y into np.arrays
	ingested by the LSTM model
	:param X: training patterns
	:param y: output patterns
	:uniq_count: number of unique elements
	:seq_length: length of patterns
	"""
	X = np.reshape(X, (len(X), seq_length, 1))
	X = X / float(uniq_count)
	# append all unique elements to y, convert to binary categorical matrix
	# then remove the rows added. This will ensure equal size and order across
	# all batches
	missing = [n for n in range(0,uniq_count+1) if n not in y]
	y_add = y + missing
	y_cat = np_utils.to_categorical(y_add)
	y_cat = y_cat[:len(y_cat)-len(missing)]

	return X, y_cat

def BatchGenerator(files):
	"""
	BatchGenerator() outputs normalized data from training json
	to be run through fit_generator to train in batches each epoch
	:param files: name of files to be ingested
	"""
	for file in files:
		with open(file) as f:
			data = json.load(f)
		X, y, uniq_count, seq_length = returnData(data)
		X_chunks = [X[n:n+32] for n in range(0,len(X), 32)]
		y_chunks = [y[n:n+32] for n in range(0,len(y), 32)]
		for x_c, y_c in zip(X_chunks, y_chunks):
			X_norm, y_norm = normalize(x_c,y_c,uniq_count,seq_length)
			yield(X_norm,y_norm)

def getDim(file):
	"""
	getDim() outputs the vector dimensions for X and y
	:param file: file to import
	"""
	with open(file) as f:
		data = json.load(f)

	X, y, count, seq_length = returnData(data)
	X_norm, y_norm = normalize(X,y,count, seq_length)

	return X_norm.shape[1], X_norm.shape[2], y_norm.shape[1]

def returnData(data):
	return data.get("X"), data.get("y"), data.get("unique_count"), data.get("seq_length")

def defineLSTM(X_1, X_2, y_1):

	# define model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X_1, X_2), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y_1, activation='softmax'))
	return model