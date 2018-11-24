from ResNet50 import ResNet50
import sys

def train_new_model():
    print("Training new model under alias recent_model")
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    model = ResNet50(input_shape = (64, 64, 3), classes = 6)

    model.load_data_h5("../../../Practice_Data/")

    model.train_model(epochs, batch_size)

    model.evaluate_model()

    model.save_model(model = "recent_model")

    del model

def update_model():
    source_model_alias = sys.argv[2]
    destination_model_alias = sys.argv[3]
    previous_loss = float(sys.argv[4])
    previous_accuracy = float(sys.argv[5])

    # Initialize a ResNet50 model to use in evaluation
    model = ResNet50(input_shape = (64, 64, 3), classes = 6)

    # Load a model given a source model alias
    model.load_model(source_model_alias)

    # Load the test and train data into the model
    model.load_data_h5(".." + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "Practice_Data" + os.path.sep)

    new_loss, new_accuracy = model.evaluate_model()

    if new_loss < previous_loss and new_accuracy > previous_accuracy:
        print("New model is better than previous best for given alias.  Saving model under alias.", destination_model_name)
        model.save_model(destination_model_name)
    else:
        print("Previous model is superior.  Can't say we didn't try : )")

    del model


if __name__ == "__main__":
    if(sys.argv[1] == '1'):
        train_new_model()
    elif(sys.argv[1] == '2'):
        update_model()
