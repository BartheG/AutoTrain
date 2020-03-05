import unittest
import sys
from unittest.mock import patch

from main import initGraph as ig
from main import main as m


class ArgumentsError(unittest.TestCase):
	def test_invalidargs(self,):
		sysargs=[ './main.py','./NotFound.MOV','./Data/' ]
		with patch.object(sys,'argv',sysargs):
			with self.assertRaises(FileNotFoundError):
				m()

	def test_badfrozengraph(self,):
		with self.assertWarns( RuntimeWarning ):
			ig( './Data/IMG_5510.MOV' )

	def test_fnffrozengraph(self,):
		from tensorflow.python.framework.errors_impl import NotFoundError
		with self.assertRaises( NotFoundError ):
			ig('./NotExits')

	def test_goodfrozengraph(self,):
		self.assertIsNotNone( ig( './Data/frozen_inference_graph.pb' ) )

if __name__ == "__main__":
	unittest.main()