import unittest
from xmltocsv import ReadAndConvert
import os

class CCutTest(unittest.TestCase):
	def test_invalidInputDir(self,):
		with self.assertRaises(FileNotFoundError):
			ReadAndConvert('DirNotFound')

	def test_validInputDir(self,):
		self.assertEqual(
			ReadAndConvert,
			type( ReadAndConvert('./TestData/') )
		)

	def test_creationoflabeldatacsv(self,):
		ReadAndConvert('./TestData/').run()
		self.assertTrue(os.path.exists('labeldata.csv'))

	def test_validlabelmap(self,):
		self.assertIsNone(
			ReadAndConvert('./TestData/').run()
		)

	def test_readvalidxml(self,):
		myParser=ReadAndConvert('./TestData')
		self.assertTrue(myParser.read())

	def test_readnotvalidxml(self,):
		myParser=ReadAndConvert('./TestData/EmptyDir/')
		self.assertFalse(myParser.read())

if __name__ == "__main__":
	unittest.main()