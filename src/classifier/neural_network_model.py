import tensorflow as tf

from data_utilities import convert_to_one_hot

class DeepNeuralNetwork(object):
	def __init__(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes):
		self.train_matrix, self.train_targets, self.test_parameter_matrix, self.test_matrix = self.flatten_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
		self.classes = classes.reshape((classes.shape[0],1))
		self.parameters = self.initialize_parameters()

	def print_dataset_shapes(self):
		print("===== Printing dataset shapes =====")
		print(self.train_matrix.shape, "train_parameter_matrix")
		print(self.train_targets.shape, "train_targets")
		print(self.test_matrix.shape, "test_parameter_matrix")
		print(self.test_targets.shape, "test_targets")
		print(self.classes.shape, "classes")

	def build_model(self, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
		"""
		Created By: Jacob Taylor Cassady
		Last Updated: 2/7/2018
	    Objective: Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

		Arguments:
		learning_rate -- learning rate of the optimization
		num_epochs -- number of epochs of the optimization loop
		minibatch_size -- size of a minibatch
		print_cost -- True to print the cost every 100 epochs		

		Returns:
		parameters -- parameters learnt by the model. They can then be used to predict.
		"""
		(n_x, m) = self.train_matrix.shape    # (n_x: input size, m : number of examples in the train set)
		n_y = train_targets.shape[0]		  # n_y : output size.
		costs = []							  # Used to keep track of varying costs.



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

	def create_placeholders(self, n_x, n_y):
		"""
		Created By: Jacob Taylor Cassady
		Last Updated: 2/8/2018
		Objective: Creates the placeholders for the tensorflow session.  These are used in the Tensorflow Graph.
    
		Arguments:
		n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
		n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
		Returns:
		X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
		Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
		Note:
		- We will use None because it let's us be flexible on the number of examples you will for the placeholders.
			In fact, the number of examples during test/train is different.
		"""

		X = tf.placeholder(shape=[n_x, None], dtype=tf.float32, name='X')
		Y = tf.placeholder(shape=[n_y, None], dtype=tf.float32, name='Y')

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

	def forward_propagation(input_matrix, parameters):
		"""
		Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
		Arguments:
		input_matrix -- input dataset placeholder, of shape (input size, number of examples)
		parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
					  the shapes are given in initialize_parameters

		Returns:
		Z3 -- the output of the last LINEAR unit
		"""
		# Retrieve the parameters from the dictionary "parameters" 
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']
		W3 = parameters['W3']
		b3 = parameters['b3']

		Z1 = tf.matmul(W1, X) + b1                                              # Z1 = np.dot(W1, X) + b1
		A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
		Z2 = tf.matmul(W2, A1) + b2                                              # Z2 = np.dot(W2, a1) + b2
		A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
		Z3 = tf.matmul(W3, A2) + b3                                              # Z3 = np.dot(W3,A2) + b3

		return Z3

	def compute_cost(Z3, targets):
		"""
		Computes the cost
	
		Arguments:
		Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
		Y -- "true" labels vector placeholder, same shape as Z3
			Returns:
		cost - Tensor of the cost function
		"""
	
		# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
		logits = tf.transpose(Z3)
		labels = tf.transpose(Y)
	
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

		return cost