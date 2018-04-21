from data_utilities import load_practice_dataset
from neural_network_model import DeepNeuralNetwork
import matplotlib.pyplot as plt

import numpy as np

def display_image(data_set, index):
    plt.imshow(data_set[index])
    plt.show()

def show_some_images(train_images, test_images):
    display_image(test_images, 0)
    display_image(train_images, 1)

    display_image(test_images, 90)
    display_image(train_images, 723)


    display_image(test_images, 7)
    display_image(train_images, 1000)


def save_model(parameters, accuracies):
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


def load_model():
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

    return parameters, accuracies

def main():
    # Load Data Set
    print("Loading practice data set.")

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()

    # Show some images (optional of course : ) )
#    show_some_images(X_train_orig, X_test_orig)

    # Flatten images and convert targets using one hot encoding

    test_model = DeepNeuralNetwork(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes)

    for i in range(5):
        new_parameters, new_accuracies = test_model.train(num_epochs = 1500, print_cost = False)
    
        old_parameters, old_accuracies = load_model()

        if (new_accuracies['train_accuracy'] >= old_accuracies['train_accuracy'] and new_accuracies['test_accuracy'] >= old_accuracies['test_accuracy']):
            print("\n\tNew model is better... Updated Accuracies: ")
            print("\ttrain_accuracy", new_accuracies['train_accuracy'])
            print("\ttest_accuracy", new_accuracies['test_accuracy'])
            save_model(new_parameters, new_accuracies)
        else:
            print("Previous model is superior.")


    parameters, accuracies = load_model()
    print("\n\n PARAMETERS: ", parameters)
    print("\n\n ACCURACIES: ", accuracies)


if __name__ == "__main__":
    main()