from SIMON import SIMON
from tkinter.filedialog import askopenfilename
import serial

""" Initialize objects. """
simon = SIMON()
com_port = input('Where is the machine I am controlling?')

def menu_prompt():
    user_response = input("\n\nWhat would you like to do?\n\t1) Improve prediction model.\n\t2) Apply the prediction model to an image.\n\t0) exit.\n\t")
    menu(user_response)

def menu(user_response):
    if user_response == '1':
        simon.improve_prediction_model(epochs = 5)
    elif user_response == '2':
        prediction = simon.predict_image(image_location = askopenfilename())
        print('writing to serial: ', bytes(prediction)[0])
        with serial.Serial(com_port, 9600) as ser:
            ser.write(bytes(prediction)[0])
    elif user_response == '0':
        exit()
    else:
        print('Invalid response please try again.')

if __name__ == '__main__':
    while(1):
        menu_prompt()