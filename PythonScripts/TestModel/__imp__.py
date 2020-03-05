import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile

import warnings
warnings.simplefilter(
	action='ignore',
	category=(FutureWarning,DeprecationWarning)
)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

from object_detection.utils import ops as utils_ops

sys.path.append('../models/research/object_detection')
from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2

PATH_TO_LABELS='./Data/labelmap.pbtxt'