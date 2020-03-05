"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""

import tensorflow as tf
import sys
import re
import os.path
from os import path
from google.protobuf import text_format
import json

genRecordFile = "generate_tfrecord"

def check_param():
	if len(sys.argv) != 3:
		raise ValueError('Invalid number of argument')
	if not path.exists(sys.argv[1]):
		raise ValueError('First param error')
	if not path.exists(sys.argv[2]):
		raise ValueError('Second param error')
	if genRecordFile not in sys.argv[1]:
		raise ValueError('First param is not a record generateur file.')
	if ".pbtxt" not in sys.argv[2]:
		raise ValueError('Second param must be a .pbtxt file')

def parse_label(label):
	res = re.findall('name: "(\w+)"\n  id: (\d)', label)
	if len(res) == 0:
		raise ValueError('Cant read label file')
	return res

def get_label():
	with tf.io.gfile.GFile(sys.argv[2], 'r') as fid:
		label_map = fid.read()
		#print(label_map)
		try:
			label = parse_label(label_map)
		except ValueError as err:
			print(err.args)
			return ''
		return label

def update_label(new_label):
	f = open(sys.argv[1], "r+")
	code = f.read()
	f.close()
	begin = code.find("def class_text_to_int(row_label):")
	end = code.find("None") + 4
	new_code = "def class_text_to_int(row_label):\n"
	for each in new_label:
		new_code = new_code + "    if row_label == \"" + each[0] + "\":\n"
		new_code = new_code + "        return  " + each[1] + "\n"
	new_code = new_code + "    else:\n        None"
	#print(code[begin:end])
	code = code.replace(code[begin:end], new_code)
	f = open(sys.argv[1], "w+")
	f.write(code)
	f.close()

def class_text_to_int(row_label):
    if row_label == "bike":
        return  1
    if row_label == "avion":
        return  4
    if row_label == "people":
        return  2
    if row_label == "car":
        return  3
    else:
        None

def main():
	try:
		check_param()
	except ValueError as err:
		print(err.args)
	new_label = get_label()
	if new_label == '':
		print('Erreur')
	else:
		update_label(new_label)
	class_text_to_int("bike")

if __name__ == "__main__":
	main()