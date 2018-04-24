from SIMON import SIMON

simon = SIMON()

def menu_prompt():
    user_response = input("\n\nWhat would you like to do?\n\t1) Improve prediction model\n\t2) Apply the prediction model to an image\n")
    menu(user_response)

def menu(user_response):
    if user_response == '1':
        simon.improve_prediction_model(epochs = 5)
    elif user_response == '2':
        simon.predict(image_location = None)
    else:
        print('Invalid response please try again.')

if __name__ == '__main__':
    while(1):
        menu_prompt()