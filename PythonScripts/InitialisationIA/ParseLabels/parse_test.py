import unittest
from parse import ParseXml
import os

class CCutTest(unittest.TestCase):
	def test_invalidInputDir(self,):
		with self.assertRaises(FileNotFoundError):
			ParseXml('DirNotFound')

	def test_validInputDir(self,):
		self.assertEqual(
			ParseXml,
			type( ParseXml('./TestData/') )
		)

	def test_creationoflabelmap(self,):
		ParseXml('./TestData/').run()
		self.assertTrue(os.path.exists('labelmap.pbtxt'))

	def test_validlabelmap(self,):
		self.assertIsNone(
			ParseXml('./TestData/').run()
		)

	def test_readvalidxml(self,):
		myParser=ParseXml('./TestData')
		self.assertTrue(myParser.read())

	def test_readnotvalidxml(self,):
		myParser=ParseXml('./TestData/EmptyDir/')
		self.assertFalse(myParser.read())

if __name__ == "__main__":
	unittest.main()