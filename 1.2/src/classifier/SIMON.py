
class SIMON(object):
    def __init__(self):
        pass

    def display_menu(self):
        print("Hello.  My name is SIMON.  I am a neural network designed to classify representations of american sign language.")
        print("1 ) Load a previously trained model.")
        print("2 ) Train a new model.")
        print("3 ) predict an image.")
        print("0 ) quit.")

        return input("What would you like me to do : ")

    def loop_menu(self):
        response = 1
        while(response != 0):
            response = int(self.menu())

if __name__ == "__main__":
    simon = SIMON()

    simon.loop_menu()
