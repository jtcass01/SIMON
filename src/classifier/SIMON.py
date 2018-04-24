from prediction_model import PredictionModel
from deep_neural_network import DeepNeuralNetwork
from data_utilities import load_practice_dataset

import numpy as np
import scipy
from scipy import ndimage

class SIMON(object):
    def __init__(self):
        self.prediction_model = self.load_model()
        print("Hello, my name is SIMON :-).  I am a Sign Integrated Machine Operating Network.  ")

    def predict_image(self, image_location):
        image = NoneType.array(ndimage.imread(image_location, flatten = False))
        X = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
        prediction = self.prediction_model.predict(X)
        print('Image prediction = ', prediction)

    def load_model(self):
        parameters = {
            'W1' : np.load('../../prior_best/paramW1.npy'),
            'b1' : np.load('../../prior_best/paramb1.npy'),
            'W2' : np.load('../../prior_best/paramW2.npy'),
            'b2' : np.load('../../prior_best/paramb2.npy'),
            'W3' : np.load('../../prior_best/paramW3.npy'),
            'b3' : np.load('../../prior_best/paramb3.npy')
        }

        accuracies = {
            'train_accuracy' : np.load('../../prior_best/trainaccuracy.npy'),
            'test_accuracy' : np.load('../../prior_best/testaccuracy.npy')
        }

        return PredictionModel(parameters, accuracies)


    def improve_prediction_model(self, epochs = 5):
        # Load Data Set
        print("Loading data set.")

        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()

        # Show some images (optional of course : ) )
    #    show_some_images(X_train_orig, X_test_orig)

        test_model = DeepNeuralNetwork(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes)

        for i in range(epochs):
            parameters, accuracies = test_model.train(num_epochs = 1500, print_cost = True)
            new_model = PredictionModel(parameters, accuracies)

            if  new_model > self.prediction_model :
                print("\n\tNew model is better... Displaying accuracies and updating files.. ")
                self.prediction_model = new_model
                self.save_model()
            else:
                print("Previous model is superior or equivalent.")

        print(self.prediction_model)


    def save_model(self):
        parameters = self.prediction_model.parameters
        accuracies = self.prediction_model.accuracies

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