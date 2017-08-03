import sys
import os
import json
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
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
	y = np_utils.to_categorical(y)
	return X, y

def BatchGenerator(files):
	"""
	BatchGenerator() outputs normalized data from training json
	to be run through fit_generator to train in batches each epoch
	:param files: name of files to be ingested
	"""
	for file in files:
		with open(file) as f:
			data = json.load(f)
		X, y, count, seq_length = returnData(data)
		X_norm, y_norm = normalize(X,y,count, seq_length)
		yield(X_norm,y_norm)

def getDim(file):
	"""
	getDim() outputs the vector dimensions for X and y
	:param file: file to import
	"""
	with open(file) as f:
		data = json.load(f)

	X, y, count, seq_length = data.get("X"), data.get("y"), data.get("unique_count"), data.get("seq_length")
	X_norm, y_norm = normalize(X,y,count, seq_length)

	return X_norm.shape[1], X_norm.shape[2], y_norm.shape[1]

def returnData(data):
	return data.get("X"), data.get("y"), data.get("unique_count"), data.get("seq_length")


def main():

	# set paths
	TRAIN_PATH = "data/training/"

	if len(sys.argv) > 1:

		# set training type
		train_type = sys.argv[1]

		# set epoch length
		if len(sys.argv) > 2:
			epoch = int(sys.argv[2])
		else:
			epoch = 100

		# import training data
		files = [TRAIN_PATH + f for f in os.listdir(TRAIN_PATH) if ".json" in f and train_type in f]

		# get training dimensions to feed LSTM
		X_1, X_2, y_1 = getDim(files[0])

		"""
		Define and compile LSTM model
		"""

		# define model
		model = Sequential()
		model.add(LSTM(256, input_shape=(X_1, X_2), return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(256))
		model.add(Dropout(0.2))
		model.add(Dense(y_1, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam')

		# define checkpoint format for when model weights are cached
		# set checkpoint so model weights are cached for an epoch
		# only when there is a loss improvement

		filepath="weights-improvement-%s{epoch:02d}-{loss:.4f}.hdf5" % train_type
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]

		# fit the model
		model.fit_generator(BatchGenerator(files), 
			steps_per_epoch=2, 
			epochs=epoch, 
			callbacks=callbacks_list,
			use_multiprocessing=True)
	else:
		raise Exception(
			"""
			You must pass one of three values in the command line:\n
			char_list, word_list, word_pos
			"""
		)

if __name__ in "__main__":
	main()