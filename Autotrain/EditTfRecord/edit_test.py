import unittest,sys
from unittest.mock import patch
import os

class CGenTFRTest(unittest.TestCase):
	def test_checkwithfilenotfound(self,):
		sysargs=[ 'test_data/generate_tfrecord_copy.py', 'teste_data/labelmap.pbtxt' ]
		with patch.object(sys,'argv',sysargs):
			with self.assertRaises(FileNotFoundError):
				if not os.path.exists(sysargs[0]) or not os.path.exists(sysargs[1]):
					raise FileNotFoundError

	def test_checkwithfileok(self,):
		sysargs=[ 'test_data/generate_tfrecord_copy.py', 'teste_data/labelmap_error.pbtxt' ]
		with patch.object(sys,'argv',sysargs):
			try:
				f = open(sysargs[0], "r")
				content = f.read()
				if "def class_text_to_int(row_label):" not in content:
					raise FileExistsError
			except:
				raise FileExistsError

if __name__ == "__main__":
	unittest.main()