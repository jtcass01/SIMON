from ResNet50 import ResNet50
import sys

if __name__ == "__main__":
    print("Training new model under alias recent_model")
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])

    model = ResNet50(input_shape = (64, 64, 3), classes = 6)
    model.load_data_h5("../../../Practice_Data/")
    model.train_model(epochs, batch_size)
    model.evaluate_model()
    model.save_model(model = "recent_model")
