import logging
import os
import sys

logger = logging.getLogger(__name__)

import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils

class SimpleCNN(object):

	def __init__(self, lr, num_epochs, beta, batch_size):
		self.learning_rate = lr
		self.num_epochs = num_epochs
		self.beta = beta
		self.batch_size = batch_size
		self.save_dir = "saves"
		self.logs_dir = "logs"
		os.makedirs(self.save_dir, exist_ok=True)
		os.makedirs(self.logs_dir, exist_ok=True)
		self.save_path = os.path.join(self.save_dir, "simple_cnn")
		self.logs_path = os.path.join(self.logs_dir, "simple_cnn")

		