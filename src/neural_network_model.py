
class DeepNeuralNetwork(object):
	def __init__(self, train_parameter_matrix, train_targets, test_parameter_matrix, test_targets):
		print(train_parameter_matrix.shape, train_targets.shape, test_parameter_matrix.shape, test_targets.shape)
