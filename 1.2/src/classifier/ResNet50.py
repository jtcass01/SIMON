import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import SVG
import scipy.misc
import os

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, model_from_json
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from resnet_utils import *

class ResNet50(object):
    """
	Author: Jacob Taylor Cassady

    Description -- Notes Taken from Professor Ng's Convolutoinal Neural Network Course on Coursera:
	Implementation of the popular ResNet50 the following architecture:
	CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
	-> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

	Arguments:
	input_shape -- shape of the images of the dataset
	classes -- integer, number of classes
    """
    def __init__(self, input_shape = (64, 64, 3), classes = 6):
        tf.reset_default_graph()

        print("\nBuilding ResNet50 model with input_shape:", str(input_shape), "and classes", str(classes))
        self.model = self.build_model(input_shape, classes)

        print("\tCompiling model with the following parameters:")
        print("\t\tOptimizer [Flavor of Gradient Descent] : Adam")
        print("\t\tLoss Function : Categorical Cross Entropy")
        print("\t\tMetrics : Accuracy")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print("\tResNet50 model now ready to load data.")
        self.X_train_orig = None
        self.Y_train_orig = None
        self.X_test_orig = None
        self.Y_test_orig = None
        self.classes = None

    def __del__(self):
        del self.model
        del self.X_train_orig
        del self.Y_train_orig
        del self.X_test_orig
        del self.Y_test_orig
        del self.classes
        del self

    def identity_block(self, X, f, filters, stage, block):
        """
        Implementation of the identity block as defined in 1.2/reference_images/identity_block.png

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X_shortcut, X])
        X = Activation('relu')(X)

        return X

    def convolutional_block(self, X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block as defined in 1.2/reference_images/convolutional_block.png

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X


        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X


    def build_model(self, input_shape, classes):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)


        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3 (≈4 lines)
        X = self.convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = self.convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = self.convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')

        return model


    def save_model(self, model = 'best_model'):
        # Save the model to JSON
        json_model = self.model.to_json()
        with open(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".json", "w") as json_file:
            json_file.write(json_model)

        # Save weights
        self.model.save_weights(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".h5")
        print("Saved model " + model + " to disk")


    def load_model(self, model = "best_model"):
        print("Attemping to load the model: " + model + " from disk.")

        # read in the model from json
        json_file = open(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".json", 'r')
        loaded_json_model = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_json_model)

        #load weights into new model
        self.model.load_weights(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".h5")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Successfully loaded model " + model + " from disk.")



    def load_data_h5(self, relative_directory_path):
        print("\nLoading data from relative directory path:", relative_directory_path)
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, self.classes = load_dataset(relative_directory_path)

        # Normalize image vectors
        self.X_train = X_train_orig/255.
        self.X_test = X_test_orig/255.

        # Convert training and test labels to one hot matrices
        self.Y_train = convert_to_one_hot(Y_train_orig, 6).T
        self.Y_test = convert_to_one_hot(Y_test_orig, 6).T

        print ("\tnumber of training examples = " + str(self.X_train.shape[0]))
        print ("\tnumber of test examples = " + str(self.X_test.shape[0]))
        print ("\tX_train shape: " + str(self.X_train.shape))
        print ("\tY_train shape: " + str(self.Y_train.shape))
        print ("\tX_test shape: " + str(self.X_test.shape))
        print ("\tY_test shape: " + str(self.Y_test.shape))

    def train_model(self, epochs = 2, batch_size = 32):
        print("\nTraining model... for ", epochs, "epochs with a batch size of", batch_size)
        self.model.fit(self.X_train, self.Y_train, epochs = epochs, batch_size = batch_size)

    def evaluate_model(self):
        print("\nEvaluating Model...")
        preds = self.model.evaluate(self.X_test, self.Y_test, verbose=1)
        print ("\tLoss = " + str(preds[0]))
        print ("\tTest Accuracy = " + str(preds[1]))
        return preds[0], preds[1]

    def predict_image(self, image_path):
        print("prepareing to predict image:", image_path)
        img = image.load_img(image_path, target_size(64, 64))

        if img is None:
            print("Unable to open image")
            return None

        pixels = image.img_to_array(img)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = preprocess_input(pixels)

        return self.model.predict(pixels)

def test_ResNet50(epochs = 2, batch_size = 32):
    test_model = ResNet50(input_shape = (64, 64, 3), classes = 6)
    test_model.load_data_h5("../../../Practice_Data/")
    test_model.train_model(epochs, batch_size)
    test_model.evaluate_model()


if __name__ == "__main__":
    test_ResNet50(2, 32)
