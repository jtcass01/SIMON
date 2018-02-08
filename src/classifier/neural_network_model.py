import tensorflow as tf

from data_utilities import convert_to_one_hot

class DeepNeuralNetwork(object):
	def __init__(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes):
		self.train_parameter_matrix, self.train_targets, self.test_parameter_matrix, self.test_targets = self.flatten_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
		self.classes = classes.reshape((classes.shape[0],1))
		self.parameters = self.initialize_parameters()

	def print_dataset_shapes(self):
		print("===== Printing dataset shapes =====")
		print(self.train_parameter_matrix.shape, "train_parameter_matrix")
		print(self.train_targets.shape, "train_targets")
		print(self.test_parameter_matrix.shape, "test_parameter_matrix")
		print(self.test_targets.shape, "test_targets")
		print(self.classes.shape, "classes")

	def flatten_data(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
		"""
		Created By: Jacob Taylor Cassady
		Last Updated: 2/7/2018
		Objective: Load in practice data from example tensorflow model.
    
		Arguments: 
		train_set_x_orig -- A NumPy array of (currently) 1080 training images of shape (64,64,3).  Total nparray shape of (1080,64,64,3)
		train_set_y_orig -- A NumPy array of (currently) 1080 training targets.  Total nparray shape of (1, 1080) [After reshape]
		test_set_x_orig -- A NumPy array of (currently) 120 test images of shape (64,64,3).  Total nparray shape of (120,64,64,3)
		test_set_y_orig -- A NumPy array of (currently) 120 test targets.  Total nparray shape of (1,120) [After reshape]
    
		Returns: 
		X_train -- A NumPy array of training data.  [Practice shape = (12288, 1080)]
		Y_train -- A NumPy array of training targets.  [Practice shape = (6, 1080)]
		X_train -- A NumPy array of test data.  [Practice shape = (12288, 120)]
		Y_train -- A NumPy array of test targets.  [Practice shape = (6, 120)]
		"""
		# Flatten the training and test images
		X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
		X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
		# Normalize image vectors
		X_train = X_train_flatten/255.
		X_test = X_test_flatten/255.
		# Convert training and test labels to one hot matrices
		Y_train = convert_to_one_hot(Y_train_orig, 6)
		Y_test = convert_to_one_hot(Y_test_orig, 6)


		return X_train, Y_train, X_test, Y_test

	def initialize_parameters(self, N1 = 25, N2 = 12):
		"""
		Initializes parameters to build a neural network with tensorflow. The shapes are:
							W1 : [N1, X_train.shape[0]]
							b1 : [N1, 1]
							W2 : [N2, N1]
							b2 : [N2, 1]
							W3 : [classes.shape[0], N2]
							b3 : [classes.shape[0], 1]
    
		Returns:
		parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
		"""

		W1 = tf.get_variable('W1', [N1, self.train_parameter_matrix.shape[0]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
		b1 = tf.get_variable('b1', [N1, 1], initializer = tf.zeros_initializer())
		W2 = tf.get_variable('W2', [N2,N1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
		b2 = tf.get_variable('b2', [N2, 1], initializer = tf.zeros_initializer())
		W3 = tf.get_variable('W3', [self.classes.shape[0],N2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
		b3 = tf.get_variable('b3', [self.classes.shape[0], 1], initializer = tf.zeros_initializer())

		parameters = {
			"W1" : W1,
			"b1" : b1,
			"W2" : W2,
			"b2" : b2,
			"W3" : W3,
			"b3" : b3,
			}

		return parameters
