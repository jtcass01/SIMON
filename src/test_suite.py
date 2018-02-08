from data_utilities import load_practice_dataset
import matplotlib.pyplot as plt


def main():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()

    plt.imshow(X_train_orig[0])

    print("Testie")



if __name__ == "__main__":
    main()