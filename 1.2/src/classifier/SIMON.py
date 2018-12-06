import os

from ResNet50 import ResNet50
from FileSystem import FileSystem

class SIMON(object):
    def __init__(self):
        self.test_loss = 100.0
        self.test_accuracy = 0.0
        self.dnn_model = None

    def update_model(self, source_model_name, destination_model_name):
        # Create a new process to update the model.  I'm doing this because I've been having trouble with pythons garbage collection
        print("source_model_name:", source_model_name)
        print("destination_model_name:", destination_model_name)
        print("self.test_loss:", self.test_loss)
        print("self.test_accuracy:", self.test_accuracy)
        os_call = "python model_processes.py 2 {0} {1} {2:.3f} {3:.3f}".format(source_model_name, destination_model_name, self.test_loss, self.test_accuracy)
        print("\nMaking command line call: ", os_call)
        os.system(os_call)

        self.test_loss, self.test_accuracy = FileSystem.load_evaluation(os.getcwd() + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + destination_model_name + "_evaluation.txt")

    def load_model(self, model_name):
        # Initialize a ResNet50 model to use in evaluation
        self.dnn_model = ResNet50(input_shape = (64, 64, 3), classes = 6)

        # Load a model given a source model alias
        self.dnn_model.load_model(model_name)

        # Load the test and train data into the model
        self.dnn_model.load_data_h5(".." + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "Practice_Data" + os.path.sep)

    def display_menu(self):
        print("\nHello.  My name is SIMON.  I am a neural network designed to classify representations of american sign language.")
        print("1 ) Load a previously trained model.")
        print("2 ) Train a new model or attempt to improve a previous one.")
        print("0 ) quit.")

        return input("What would you like me to do : ")

    def loop_menu(self):
        response = 1
        while(response != 0):
            response = int(self.display_menu())
            self.perform_action(menu_response = response)
            if(response == 1):
                response = 0
            print("Action Complete.\n")

    def loop_model_menu(self):
        response = 1
        while(response != 0):
            print("\nA model has been loaded.")
            print("1 ) Predict an image.")
            print("2 ) Evaluate the model.")
            print("3 ) Control a device.")
            print("0 ) Exit.")
            response = int(input('What would you like to do : '))

            if response == 0:
                print("Exiting...")
            elif response == 1:
                self.prompt_predict_image()
            elif response == 2:
                self.dnn_model.evaluate_model()
            elif response == 3:
                pass
            else:
                print("Invalid menu response.  Please try again.")


    def perform_action(self, menu_response):
        if menu_response == 0:
            print("Exiting...")
        elif menu_response == 1:
            self.prompt_load_previous_model()
            self.loop_model_menu()
        elif menu_response == 2:
            self.prompt_train_model()
        else:
            print("Invalid menu response.  Please try again.")

    def prompt_load_previous_model(self):
        model_name = input("\nWhat is the alias of the model you would like to load : ")

        if os.path.isfile(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model_name + ".h5"): # previous model exists
            self.load_model(model_name)
        else:
            print("Unable to find a previous model matching the given alias.")

    def prompt_train_model(self):
        model_name = input("\nWhat is the alias of the model you would like to train : ")

        if os.path.isfile("../../models/" + model_name + ".h5"): # previous model exists
            print("Previous model found matching given alias.  Evaluating previous model...")
            self.update_model(source_model_name=model_name, destination_model_name=model_name)
            print("Previous model has a loss of {} and an accuracy of {}".format(self.test_loss, self.test_accuracy))
            print("Maybe we can do better.")
        else:
            print("Unable to find a previous model matching the given alias.")

        attempts = int(input("Let's start training new models.  How many training attempts would you like to perform : "))
        epochs = int(input("How many epochs should each training attempt complete : "))
        batch_size = int(input("How large should each training batch be : "))

        for attempt in range(attempts):
            print("\nBeginning training attempt: ", attempt)
            # Train a new model and log it under the alias "recent_model"
            self.train_new_model(model_name, epochs, batch_size)

            # Update the evaluation of the model and overwrite the previous save if the recent model is better.
            self.update_model(source_model_name="recent_model", destination_model_name=model_name)

        print("Done with all training attempts.  You will need to reload this model using the alias {} before performing predictions.".format(model_name))

    def train_new_model(self, model_name, epochs, batch_size):
        print("\nCreating a new process to train " + str(model_name))

        # Create a new process to train a model.  I'm doing this because I've been having trouble with pythons garbage collection
        print("\nMaking command line call: ", "python model_processes.py 1 " + str(epochs) + " " + str(batch_size))
        os.system("python model_processes.py 1 " + str(epochs) + " " + str(batch_size))

    def prompt_predict_image(self):
        if self.dnn_model is None:
            print("You need to load or train a model for me to perform predictions with.")
        else:
            image_path = input("Where is your image located : ")
            prediction = self.dnn_model.predict_image(image_path = image_path)
            print("The model predicted:", prediction)


if __name__ == "__main__":
    simon = SIMON()
    simon.loop_menu()
