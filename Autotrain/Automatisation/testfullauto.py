import unittest
from main import JsonParser
from main import Process
import os,shutil

def clear():
	if 'delivery_1.0' in os.getcwd():
		os.chdir('..')
		shutil.rmtree('./delivery_1.0')

class AutomatTest(unittest.TestCase):
	def test_wrongjsonpath(self,):
		with self.assertRaises(FileNotFoundError):
			JsonParser('wrongpath.json')
		clear()

	def test_wrongjsonfile(self,):
		with self.assertRaises(TypeError):
			JsonParser('./DataTest/wrong.txt')
		clear()

	def test_wrongoutputdir(self,):
		with self.assertRaises(FileNotFoundError):
			JsonParser('./DataTest/input_badoutput.json')
		clear()

	def test_wrongmodulename(self,):
		with self.assertRaises(ModuleNotFoundError):
			Process('./DataTest/input_badmodulename.json').run()
		clear()

	def test_wrongclassname(self,):
		with self.assertRaises(AttributeError):
			Process( './DataTest/input_badclassname.json' ).run()
		clear()

	def test_wrongclassparam(self,):
		with self.assertRaises(FileNotFoundError):
			Process( './DataTest/input_badclassparam.json' ).run()
		clear()

	def test_wrongscriptpath(self,):
		self.assertEqual(False,
		Process('./DataTest/input_badscriptpath.json').run())
		clear()

	def test_wrongscriptpath(self,):
		self.assertEqual(False,
		Process('./DataTest/input_badscriptargs.json').run())
		clear()

	def test_wrongscriptpath(self,):
		self.assertEqual(False,
		Process('./DataTest/input_badcommand.json').run())
		clear()

if __name__ == "__main__":
	unittest.main()