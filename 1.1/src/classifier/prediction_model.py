import tensorflow as tf

class PredictionModel(object):
    def __init__(self, parameters, accuracies):
        self.parameters = parameters
        self.accuracies = accuracies

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
        
        return prediction

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


