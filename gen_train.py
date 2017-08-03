import os
import json
import numpy as np

# set paths
INPATH = "data/book_json/"
OUTPATH = "data/training/%s"

# set element types
ELEMENTS = {
	'char_list': 100,
	'word_list': 20,
	'word_pos': 20
}

def generateTrain(data, element_type, seq_length):

	# generate one list holding all book's elements
	el_all = [g for group in [d.get(element_type) for d in data] for g in group]

	# reduce to list of all unique elements
	el_uniq = sorted(list(set(el_all)))

	# store counts
	all_count = len(el_all)
	uniq_count = len(el_uniq)

	print(all_count)
	print(uniq_count)

	# create mapping of unique characters, reverse mapping
	el_to_int = dict((e, i) for i, e in enumerate(el_uniq))
	int_to_el = dict((i, e) for i, e in enumerate(el_uniq))

	# generate character sequences to train NN
	X, y = genIntSeq(el_all, el_to_int, seq_length)
	print(len(X))

	"""
	return {
		'X': X,
		'y': y,
		'element_count': all_count,
		'unique_count': uniq_count,
		'element_to_int': el_to_int,
		'int_to_element': int_to_el,
		'seq_lenth': seq_length,
		'elements': el_all,
		'unique_elements': el_uniq
	}
	"""

def storeBooks(files):
	results = []
	for file in files:
		with open(file) as f:
			results.append(json.load(f))
	return results

def genIntSeq(elements, el_2_int, seq_length):
	dataX = []
	dataY = []
	for n in range(0, len(elements) - seq_length, 1):
		seq_in = elements[n:n+seq_length]
		seq_out = elements[n + seq_length]
		dataX.append([el_2_int[e] for e in seq_in])
		dataY.append(el_2_int[seq_out])
	return dataX, dataY

def main():
	"""
	Prepare training data for LSTM Neural Net
	"""
	# import training data
	files = [INPATH + f for f in os.listdir(INPATH) if ".json" in f]

	# import books
	print("Loading training data...")
	bookList = storeBooks(files)
	print("Load complete.")

	# loop through element types and store data structure
	for key, value in ELEMENTS.items():
		print("Generating: %s" % key)
		outfile = "%s.json" % key
		generateTrain(bookList, key, value)
		#with open(OUTPATH % outfile, 'w') as f:
			#json.dump(generateTrain(bookList, key, value), f)

if __name__ in "__main__":
	main()

	
