from data_utilities import load_practice_dataset
import matplotlib.pyplot as plt

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()


def display_image(data_set, index):
    plt.imshow(data_set[index])
    plt.show()

def show_some_images():
    display_image(X_test_orig, 0)
    display_image(X_train_orig, 1)

    display_image(X_test_orig, 90)
    display_image(X_train_orig, 723)


    display_image(X_test_orig, 7)
    display_image(X_train_orig, 1000)


def main():
    show_some_images()

    print("Testie")



if __name__ == "__main__":
    main()