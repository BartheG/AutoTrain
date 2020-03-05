import unittest
from convimg import ConvImg
import os

class CConvImgTest(unittest.TestCase):
	def test_invalidInputDir(self,):
		with self.assertRaises(FileNotFoundError):
			ConvImg('DirNotFound','300*300')

	def test_validInputDir(self,):
		self.assertEqual(
			ConvImg,
			type( ConvImg('./TestDirectory/','300*300') )
		)

	def test_invalidsize(self,):
		with self.assertRaises(ValueError):
			ConvImg('./TestDirectory','InvalidSize')
		with self.assertRaises(ValueError):
			ConvImg('./TestDirectory','300*')
		with self.assertRaises(ValueError):
			ConvImg('./TestDirectory','*300*')
		with self.assertRaises(ValueError):
			ConvImg('./TestDirectory','100000*300')
		with self.assertRaises(ValueError):
			ConvImg('./TestDirectory','-1*8')
		with self.assertRaises(ValueError):
			ConvImg('./TestDirectory','0*0')

	def test_creationofresultdirectory(self,):
		ConvImg('./TestDirectory/','300*300')
		self.assertTrue(os.path.exists('converted_300*300_img'))

	def test_validconvertedimg(self,):
		self.assertIsNone(
			ConvImg('./TestDirectory/','300*300').run()
		)
		for path in [ 'bird.jpeg','circle.jpg','moon.jpg' ]:
			self.assertTrue(os.path.exists( './converted_300*300_img/'+path ))
		for path in [ 'breson.jpg','trace.txt' ]:
			self.assertFalse(os.path.exists( './converted_300*300_img/'+path ))


if __name__ == "__main__":
	unittest.main()