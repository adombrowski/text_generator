import re
import nltk
import os
import json

"""
genBookJSON.py restructures .txt from Project Gutenberg
into json dictionary:

{
	'source_path': path to original .txt,
	'text': document's main body text parsed,
	'elements':[
		'char_list': list holding every character in order,
		'word_list': list of all words present in document,
		'word_pos': list holding tuple containing word and it's part of speech
	]
}

"""

# set paths
TEXT_PATH = "data/texts/"
JSON_PATH = "data/book_json/"

def parse(raw_text, file_name):

	"""
	parse() extracts and restructures gutenberg raw
	text into dictionary output
	:param raw_text: text from .txt file
	:file_name: original .txt file name
	"""

	# set regex
	ANCHOR = "\*\*\*[^\*]+(Project Gutenberg|PROJECT GUTENBERG)[^\*]+\*\*\*"

	# parse document body
	text = re.search("%s.+%s" % (ANCHOR, ANCHOR), raw_text, re.DOTALL).group(0)
	text = re.sub(ANCHOR, "", text)

	# generate list of characters
	char_list = [c for c in re.sub("\n", "", text.lower())]
	
	# generate list of unique characters
	unique_char = list(set(char_list))

	# generate list of words
	word_list = [w for w in text.split()]

	# generate list of word/pos tuples
	word_pos = ["-".join(w) for w in nltk.pos_tag(word_list)]

	return {
		'source_path': file_name,
		'text': text,
		'char_list': char_list,
		'word_list': word_list,
		'word_pos': word_pos
	}

def book_2_json(files):
	"""
	book_2_json() loads files, stores data in dictionary,
	dumps json in new folder
	:param files: list of files to convert
	"""
	for file in files:
		print("Converting: %s" % file)
		with open(TEXT_PATH + file) as f:
			raw_text = f.read()

		with open(JSON_PATH + file.replace(".txt", ".json"), 'w') as f:
			json.dump(parse(raw_text, file), f)
		print("Dump Success!")

def main():

	# load text file names
	files = [f for f in os.listdir(TEXT_PATH) if ".txt" in f]

	# load file and gen dictionary entry
	# store as json in data/ folder
	book_2_json(files)

if __name__ in "__main__":
	main()




