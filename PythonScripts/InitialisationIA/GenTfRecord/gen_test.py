import unittest,sys
from unittest.mock import patch
from m_gentfr import check

class CGenTFRTest(unittest.TestCase):
	def test_checkwithfilenotfound(self,):
		sysargs=[ './m_gentfr.py','./NotFound.csv','./TestData/','./labeldata.csv','./TestData/' ]
		with patch.object(sys,'argv',sysargs):
			with self.assertRaises(FileNotFoundError):
				check()

	def test_checkwithfileok(self,):
		sysargs=[ './m_gentfr.py','./labeldata.csv','./labeldata.csv','./TestData/','./TestData/' ]
		with patch.object(sys,'argv',sysargs):
				self.assertIsNone(check())

if __name__ == "__main__":
	unittest.main()