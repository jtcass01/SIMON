from data_utilities import load_practice_dataset
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


def main():
    # Load Data Set
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()

    # Show some images (optional of course : ) )
    show_some_images(X_train_orig, X_test_orig)

    # Flatten images and convert targets using one hot encoding

    test_model = DeepNeuralNetwork(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes)



    parameters = test_model.train()


if __name__ == "__main__":
    main()