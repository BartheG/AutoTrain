import unittest
from cut import OrganizeData
import os

class CCutTest(unittest.TestCase):
	def test_invalidInputDir(self,):
		with self.assertRaises(FileNotFoundError):
			OrganizeData('DirNotFound')

	def test_validInputDir(self,):
		self.assertEqual(
			OrganizeData,
			type( OrganizeData('./TestDirectory/') )
		)

	def test_creationofresultdirectory(self,):
		OrganizeData('./TestDirectory/')
		for path in [ './test/','./train/','./test/data','./train/data','./test/labels','./train/labels' ]:
			self.assertTrue(os.path.exists(path))

	def test_validcut(self,):
		self.assertIsNone(
			OrganizeData('./TestDirectory/').run()
		)

	def test_validprop(self,):
		def findInDir(pathdir,ext):
			return [ os.path.join(root, file) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) ]
		from cut import findInDir as cutfind
		nbfilestest=0
		nbfilestrain=0
		nballfiles=0
		for ext in [ '.jpg','.jpeg','.png' ]:
			nbfilestest+=len(findInDir('./test/',ext))
			nbfilestrain+=len(findInDir('./train',ext))
			nballfiles+=len(cutfind('./TestDirectory',ext))
		#print(nballfiles,nbfilestest,nbfilestrain)
		self.assertGreater(nbfilestrain,nbfilestest*2)
		self.assertEqual( nbfilestest+nbfilestrain,nballfiles )

	def test_filewithxml(self,):
		def findInDir(pathdir,ext):
			return [ os.path.join(root, file) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) ]
		for ext in [ '.jpg','.jpeg','.png' ]:
			for file in findInDir( './test/',ext ):
				xmlpath=file.replace('data','labels').replace(ext,'.xml')
				self.assertTrue( os.path.exists(xmlpath) )

if __name__ == "__main__":
	unittest.main()