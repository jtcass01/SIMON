from SIMON import SIMON
from tkinter.filedialog import askopenfilename

simon = SIMON()

def menu_prompt():
    user_response = input("\n\nWhat would you like to do?\n\t1) Improve prediction model.\n\t2) Apply the prediction model to an image.\n\t0) exit.\n\t")
    menu(user_response)

def menu(user_response):
    if user_response == '1':
        simon.improve_prediction_model(epochs = 1)
    elif user_response == '2':
        simon.predict_image(image_location = askopenfilename())
    elif user_response == '0':
        exit()
    else:
        print('Invalid response please try again.')

if __name__ == '__main__':
    while(1):
        menu_prompt()