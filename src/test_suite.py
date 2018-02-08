from data_utilities import load_practice_dataset, convert_to_one_hot
from neural_network_model import DeepNeuralNetwork
import matplotlib.pyplot as plt

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

def flatten_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
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



def main():
    # Load Data Set
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()

    # Show some images (optional of course : ) )
#    show_some_images(X_train_orig, X_test_orig)

    # Flatten images and convert targets using one hot encoding
    train_parameter_matrix, train_targets, test_parameter_matrix, test_targets = flatten_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

    test_model = DeepNeuralNetwork(train_parameter_matrix, train_targets, test_parameter_matrix, test_targets)


if __name__ == "__main__":
    main()