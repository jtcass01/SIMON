from ResNet50 import ResNet50
import os

class SIMON(object):
    def __init__(self):
        self.test_loss = 1.0
        self.test_accuracy = 0.0
        self.dnn_model = None

    def display_menu(self):
        print("Hello.  My name is SIMON.  I am a neural network designed to classify representations of american sign language.")
        print("1 ) Load a previously trained model.")
        print("2 ) Train a new model or attempt to improve a previous one.")
        print("3 ) predict an image.")
        print("0 ) quit.")

        return input("What would you like me to do : ")

    def loop_menu(self):
        response = 1
        while(response != 0):
            response = int(self.display_menu())
            self.perform_action(menu_response = response)
            print("Action Complete.\n")

    def perform_action(self, menu_response):
        if menu_response == 0:
            print("Exiting...")
        elif menu_response == 1:
            self.prompt_load_previous_model()
        elif menu_response == 2:
            self.prompt_train_model()
        elif menu_response == 3:
            self.prompt_predict_image()
        else:
            print("Invalid menu response.  Please try again.")

    def prompt_load_previous_model(self):
        print("Loading previous model...")

    def load_previous_model(self, model_name):
        pass

    def prompt_train_model(self):
        model_name = input("What is the alias of the model you would like to train : ")

        model_path = os.path.dirname(os.path.realpath(__file__)) + model_name + ".h5"

        if os.path.isfile(model_path): # previous model exists
            print("Previous model found.")
        else:
            print("Unable to find a previous model matching the given alias.")

    def train_new_model(self, model_name):
        pass

    def prompt_predict_image(self):
        if self.dnn_model is None:
            print("You need to load or train a model for me to perform predictions with.")
        else:
            image_path = input("Where is your image located : ")
            self.dnn_model.predict_image(image_path = image_path)


if __name__ == "__main__":
    simon = SIMON()
    simon.loop_menu()
