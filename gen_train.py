import os
import json
import numpy as np

# set paths
INPATH = "data/book_json_test/"
OUTPATH = "data/training/%s"

# set batch size
BATCH_SIZE = 100000

# set element types
ELEMENTS = {
	'char_list': 100,
	'word_list': 20,
	'word_pos': 20
}

def generateTrain(data, element_type, seq_length, outpath):

	# generate one list holding all book's elements
	el_all = [g for group in [d.get(element_type) for d in data] for g in group]
	print(len(el_all))

	# reduce to list of all unique elements
	el_uniq = sorted(list(set(el_all)))
	uniq_count = len(el_uniq)

	# create mapping of unique characters, reverse mapping
	el_to_int = dict((e, i) for i, e in enumerate(el_uniq))
	int_to_el = dict((i, e) for i, e in enumerate(el_uniq))

	# initiate data dictionary
	data = {
		'element_to_int': el_to_int,
		'int_to_element': int_to_el,
		'seq_length': seq_length,
		'unique_count': uniq_count
	}

	# if # of elements < batch size, output
	if len(el_all) <= BATCH_SIZE:
		X, y = genIntSeq(elements, el_to_int, seq_length)

		# update dictionary
		out_data = updateDict(data, X=X, y=y)

		# store data
		storeData(out_data, outpath)
	else:
		batchData(el_all, data, outpath, seq_length)

def updateDict(dic, **kwargs):
	for key, value in kwargs.items():
		dic[key] = value
	return dic

def storeData(data, outpath):
	with open(outpath, 'w') as f:
		json.dump(data, f)
	
def batchData(elements, data, outpath, seq_length):
	# generate element sequences
	# to conserve space, will batch
	el_to_int = data.get("element_to_int")
	all_count = len(elements)
	loop_limit = all_count // BATCH_SIZE

	for n in range(0, loop_limit):
		upper_bound = BATCH_SIZE * (n+1)
		lower_bound = BATCH_SIZE * n
		el_sub = elements[lower_bound:upper_bound]
		X, y = genIntSeq(el_sub, el_to_int, seq_length)

		# update data
		out_data = updateDict(data, X=X, y=y)

		# append batch name to outpath
		outpath_batch = outpath.replace(".json", "_batch{}.json".format(n+1))

		# store data
		storeData(out_data, outpath_batch)

def loadBooks(files):
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
	bookList = loadBooks(files)
	print("Load complete.")

	# loop through element types and store data structure
	for key, value in ELEMENTS.items():
		print("Generating: %s" % key)

		# set file outpath
		outfile = "%s.json" % key
		outpath = OUTPATH % outfile

		generateTrain(bookList, key, value, outpath)

if __name__ in "__main__":
	main()