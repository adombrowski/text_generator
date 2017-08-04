import sys
import os
import json
import numpy as np

from keras.callbacks import ModelCheckpoint

from nn_utils import *

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
		model = defineLSTM(X_1, X_2, y_1)

		# define checkpoint format for when model weights are cached
		# set checkpoint so model weights are cached for an epoch
		# only when there is a loss improvement

		filepath="weights-improvement-%s-{epoch:02d}-{loss:.4f}.hdf5" % train_type
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]

		model.compile(loss='categorical_crossentropy', optimizer='adam')

		# fit the model
		model.fit_generator(BatchGenerator(files), 
			steps_per_epoch=10000, 
			epochs=epoch, 
			callbacks=callbacks_list,
			)
	else:
		raise Exception(
			"""
			You must pass one of three values in the command line:\n
			char_list, word_list, word_pos
			"""
		)

if __name__ in "__main__":
	main()