import unittest
from cut import MovieCut
import os

class CCutTest(unittest.TestCase):
	def test_invalidInputDir(self,):
		with self.assertRaises(FileNotFoundError):
			MovieCut('DirNotFound')

	def test_validInputDir(self,):
		self.assertEqual(
			MovieCut,
			type( MovieCut('./TestDirectory/') )
		)

	def test_creationofresultdirectory(self,):
		MovieCut('./TestDirectory/')
		self.assertTrue(os.path.exists('./exported_frame'))

	def test_validcut(self,):
		self.assertIsNone(
			MovieCut('./TestDirectory/').run()
		)

if __name__ == "__main__":
	unittest.main()