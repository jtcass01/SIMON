import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy import ndimage
from tensorflow.python.framework import ops

from data_utilities import convert_to_one_hot, random_mini_batches, load_practice_dataset


class DeepNeuralNetwork(object):
	def __init__(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes):
		tf.reset_default_graph()
		self.train_matrix, self.train_targets, self.test_matrix, self.test_targets = self.flatten_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
		self.classes = classes.reshape((classes.shape[0],1))
		self.parameters = self.initialize_parameters()

	def print_dataset_shapes(self):
		print("===== Printing dataset shapes =====")
		print(self.train_matrix.shape, "train_parameter_matrix")
		print(self.train_targets.shape, "train_targets")
		print(self.test_matrix.shape, "test_parameter_matrix")
		print(self.test_targets.shape, "test_targets")
		print(self.classes.shape, "classes")

	def train_model(self, learning_rate = 0.0001, epochs = 1500, batch_size = 32, print_cost = True):
		"""
		Created By: Jacob Taylor Cassady
		Last Updated: 2/7/2018
	    Objective: Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

		Arguments:
		learning_rate -- learning rate of the optimization
		epochs -- number of epochs of the optimization loop
		batch_size -- size of a minibatch
		print_cost -- True to print the cost every 100 epochs

		Returns:
		parameters -- parameters learnt by the model. They can then be used to predict.
		"""
		print("Entering train....")
		ops.reset_default_graph()
		(n_x, m) = self.train_matrix.shape    # (n_x: input size, m : number of examples in the train set)
		n_y = self.train_targets.shape[0]		  # n_y : output size.
		costs = []							  # Used to keep track of varying costs.

		# Create placeholders for TensorFlow graph of shape (n_x, n_y)
		print("Creating placeholders for TensorFlow graph...")
		X, Y = self.create_placeholders(n_x, n_y)
		print("Complete.\n")

		# Initialize Parameters
		print("Initailizing parameters for TensorFlow graph...")
		parameters = self.initialize_parameters()
		print("Complete.\n")

		# Build the forward propagation in the TensorFlow Graph
		print("Building the forward propagation in the TensorFlow Graph...")
		Z3 = self.forward_propagation(X, parameters)
		print("Complete.\n")

		# Add the cost function to the Tensorflow Graph
		print("Adding cost function to the TensorFlow Graph")
		cost = self.compute_cost(Z3, Y)
		print("Complete.\n")

		# Define the TensorFlow Optimizer.. We are using an AdamOptimizer.
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		# Initialize all the variables with our newly made TensorFlow Graph
		init = tf.global_variables_initializer()

		# Use the TensorFlow Graph to train the parameters.
		with tf.Session() as session:
			# Run the initialization
			session.run(init)

			# Perform Training
			for epoch in range(epochs):
				epoch_cost = 0.								# Defines a cost related to the current epoch
				num_minibatches = int(m / batch_size)	# Calculates the number of minibatches in the trainset given a minibatch size
				minibatches = random_mini_batches(self.train_matrix, self.train_targets, batch_size)

				for minibatch in minibatches:
					# Retrieve train_matrix and train_targets from minibatch
					mini_matrix, mini_targets = minibatch

					# Run the session to execute the "optimizer" and the "cost",
					_, minibatch_cost = session.run([optimizer, cost], feed_dict={X:mini_matrix, Y:mini_targets})

					# Sum epoch cost
					epoch_cost += minibatch_cost / num_minibatches

				# Done training.  Print the cost of every 100 epochs
				if print_cost == True and epoch % 100 == 0:
					print("Cost after epoch %i: %f" % (epoch, epoch_cost))
				# Keep track of the cost of every 5 epochs for plotting later
				if print_cost == True and epoch % 5 == 0:
					costs.append(epoch_cost)

			# Plot the costs for analysis
			plt.plot(np.squeeze(costs))
			plt.ylabel('cost')
			plt.xlabel('iteration ( per 5 )')
			plt.title('Learning rate = ' + str(learning_rate))
			if print_cost == True:
				#plt.show()
				pass

			# Save the parameters as a varaible for prediction and evaluation of fit to test set.
			parameters = session.run(parameters)

			# Develop TensorFlow prediction standards for testing accuracy  of test and train sets
			correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

			# Develop accuracy identifier using TensorFlow
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

			# Display accuracy of train and test predictions.
			print("Train Accuracy: ", accuracy.eval({X: self.train_matrix, Y: self.train_targets}))
			print("Test Accuracy: ", accuracy.eval({X: self.test_matrix, Y: self.test_targets}))

			# Return parameters for prediction against the model.
			self.parameters = parameters

			train_accuracy = accuracy.eval({X: self.train_matrix, Y: self.train_targets})
			test_accuracy = accuracy.eval({X: self.test_matrix, Y: self.test_targets})

			accuracies = {"train_accuracy": train_accuracy,
                          "test_accuracy" : test_accuracy}

			# Return parameters for prediction against the model.
			return parameters, accuracies

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

		return X, Y

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
		W1 = tf.get_variable('W1', [N1, self.train_matrix.shape[0]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
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

	def forward_propagation(self, input_matrix, parameters):
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

		Z1 = tf.matmul(W1, input_matrix) + b1                            # Z1 = np.dot(W1, X) + b1
		A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
		Z2 = tf.matmul(W2, A1) + b2                                      # Z2 = np.dot(W2, a1) + b2
		A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
		Z3 = tf.matmul(W3, A2) + b3                                      # Z3 = np.dot(W3,A2) + b3

		return Z3

	def compute_cost(self, Z3, targets):
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
		labels = tf.transpose(targets)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

		return cost


class DNNPredictionModel(object):
	def __init__(self, parameters, accuracies):
		self.parameters = parameters
		self.accuracies = accuracies
	def predict_image(self, image_path):
		image = np.array(ndimage.imread(ismage_path, flatten = False))
		pixels = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
		return self.predict(pixels)

	def evaluate_model(self):
		print(self)

	def predict(self, X):
		W1 = tf.convert_to_tensor(self.parameters["W1"])
		b1 = tf.convert_to_tensor(self.parameters["b1"])
		W2 = tf.convert_to_tensor(self.parameters["W2"])
		b2 = tf.convert_to_tensor(self.parameters["b2"])
		W3 = tf.convert_to_tensor(self.parameters["W3"])
		b3 = tf.convert_to_tensor(self.parameters["b3"])

		params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

		x = tf.placeholder("float", [12288, 1])

		z3 = self.forward_propagation_for_predict(x)
		p = tf.argmax(z3)

		sess = tf.Session()
		prediction = sess.run(p, feed_dict = {x: X})

		return prediction[0]

	def forward_propagation_for_predict(self, X):
		"""
		Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

		Arguments:
		X -- input dataset placeholder, of shape (input size, number of examples)
		parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
		              the shapes are given in initialize_parameters
		Returns:
		Z3 -- the output of the last LINEAR unit
		"""

		# Retrieve the parameters from the dictionary "parameters"
		W1 = self.parameters['W1']
		b1 = self.parameters['b1']
		W2 = self.parameters['W2']
		b2 = self.parameters['b2']
		W3 = self.parameters['W3']
		b3 = self.parameters['b3']
		                                                       # Numpy Equivalents:
		Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
		A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
		Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
		A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
		Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

		return Z3

	""" Overloading of string operator """
	def __str__(self):
		return "\t === CURRENT PREDICTION MODEL ===\n" + '\tTrain Accuracy: ' + str(self.accuracies['train_accuracy']) + '\n\tTest Accuracy: ' + str(self.accuracies['test_accuracy']) + '\n\tParameters: ' + str(self.parameters)

	""" Overloading of comparison operators """
	def __eq__(self, other):
		return self.accuracies['train_accuracy'] == other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] == other.accuracies['test_accuracy']

	def __le__(self, other):
		return self.accuracies['train_accuracy'] <= other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] <= other.accuracies['test_accuracy']

	def __lt__(self,other):
		return self.accuracies['train_accuracy'] > other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] > other.accuracies['test_accuracy']

	def __ge__(self, other):
		return self.accuracies['train_accuracy'] >= other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] >= other.accuracies['test_accuracy']

	def __gt__(self,other):
		return self.accuracies['train_accuracy'] > other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] > other.accuracies['test_accuracy']

	def load_model(self, model="dnn_best"):
		self.parameters = {
		    'W1' : np.load('../../models/' + model + '/paramW1.npy'),
		    'b1' : np.load('../../models/' + model + '/paramb1.npy'),
		    'W2' : np.load('../../models/' + model + '/paramW2.npy'),
		    'b2' : np.load('../../models/' + model + '/paramb2.npy'),
		    'W3' : np.load('../../models/' + model + '/paramW3.npy'),
		    'b3' : np.load('../../models/' + model + '/paramb3.npy')
		}

		self.accuracies = {
		    'train_accuracy' : np.load('../../models/' + model + '/trainaccuracy.npy'),
		    'test_accuracy' : np.load('../../models/' + model + '/testaccuracy.npy')
		}


	def improve_prediction_model(self, epochs = 5):
		# Load Data Set
		print("Loading data set.")

		X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()

		test_model = DeepNeuralNetwork(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes)

		for i in range(epochs):
			parameters, accuracies = test_model.train(num_epochs = 1500, print_cost = True)
			new_model = PredictionModel(parameters, accuracies)

			if  new_model > self.prediction_model:
				print("\n\tNew model is better... Displaying accuracies and updating files.. ")
				self.prediction_model = new_model
				print(self.prediction_model)
				self.save_model()
			else:
				print("Previous model is superior or equivalent.")

		print(self)

	def save_model(self):
		parameters = self.parameters
		accuracies = self.accuracies

		W1 = parameters['W1']
		np.save('../../prior_best/paramW1.npy',W1)

		b1 = parameters['b1']
		np.save('../../prior_best/paramb1.npy',b1)

		W2 = parameters['W2']
		np.save('../../prior_best/paramW2.npy',W2)

		b2 = parameters['b2']
		np.save('../../prior_best/paramb2.npy',b2)

		W3 = parameters['W3']
		np.save('../../prior_best/paramW3.npy',W3)

		b3 = parameters['b3']
		np.save('../../prior_best/paramb3.npy',b3)

		train_accuracy = accuracies['train_accuracy']
		np.save('../../prior_best/trainaccuracy.npy',train_accuracy)

		test_accuracy = accuracies['test_accuracy']
		np.save('../../prior_best/testaccuracy.npy',test_accuracy)


	def predict_image(self, image_path):
		image = np.array(ndimage.imread(image_path, flatten = False))
		X = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T

		return self.predict(X)
