from SIMON import SIMON
from tkinter.filedialog import askopenfilename
import serial

""" Initialize objects. """
simon = SIMON()
#com_port = input('Where is the machine I am controlling?\n')
#ser = serial.Serial(com_port, 9600)


def menu_prompt():
    user_response = input("\n\nWhat would you like to do?\n\t1) Improve prediction model.\n\t2) Apply the prediction model to an image.\n\t0) exit.\n\t")
    menu(user_response)

def menu(user_response):
    if user_response == '1':
        epochs = int(input('How many improvement attempts would you like to make?\n'))
        simon.improve_prediction_model(epochs = epochs)
    elif user_response == '2':
        prediction = simon.predict_image(image_location = askopenfilename())
        print("SIMON guessed ", prediction)
#        write_response(prediction)
    elif user_response == '0':
#        ser.close()
        exit()
    else:
        print('Invalid response please try again.')

def write_response(integer_value):
    if integer_value == 0:
        ser.write(b'0')
    if integer_value == 1:
        ser.write(b'1')
    if integer_value == 2:
        ser.write(b'2')
    if integer_value == 3:
        ser.write(b'3')
    if integer_value == 4:
        ser.write(b'4')
    if integer_value == 5:
        ser.write(b'5')


if __name__ == '__main__':
    while(1):
        menu_prompt()
