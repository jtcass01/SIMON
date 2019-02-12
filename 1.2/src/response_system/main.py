import pigpio
import time
import random
from gpiozero import Button
import serial

ser = serial.Serial(
  port='/dev/ttyS0',
  baudrate = 115200,
  parity = serial.PARITY_NONE,
  stopbits=serial.STOPBITS_ONE,
  bytesize=serial.EIGHTBITS,
  timeout=1
)

pi = pigpio.pi()

# Set pins
RED_PIN = 27
GREEN_PIN = 17
BLUE_PIN = 22

def toMaxColor(color, value):
  for i in range(1, value, 1):
    pi.set_PWM_dutycycle(color, i)
    time.sleep(0.05)

def toMinColor(color, value):
  for i in range(value, 1, -1):
    pi.set_PWM_dutycycle(color, i)
    time.sleep(0.05)

def toMaxColors(colorOne, colorTwo, value):
  for i in range(1, value, 1):
    pi.set_PWM_dutycycle(colorOne, i)
    pi.set_PWM_dutycycle(colorTwo, i)
    time.sleep(0.05)

def turnOffLEDs():
  pi.set_PWM_dutycycle(RED_PIN, 0)
  pi.set_PWM_dutycycle(GREEN_PIN, 0)
  pi.set_PWM_dutycycle(BLUE_PIN, 0)

def getValue():
  turnOffLEDs()
  net_num = random.randint(0,6)
  setValue(net_num)

def setValue(net_num):
  if net_num == 0:
    intensity = 0
  elif net_num == 1:
    intensity = 43
  elif net_num == 2:
    intensity = 86
  elif net_num == 3:
    intensity = 129
  elif net_num == 4:
    intensity = 172
  elif net_num == 5:
    intensity = 215
  else:
    intensity = 255
  return intensity

def promptForColorAndValue():
  color = raw_input("[R], [G], or [B]: ")
  value = raw_input("Enter value: ")
  value = int(value) # cast value to int
  if color is "R" and value >= 1 and value <= 255:
    pi.set_PWM_dutycycle(RED_PIN, value)
  elif color is "G" and value >= 1 and value <= 255:
    pi.set_PWM_dutycycle(GREEN_PIN, value)
  elif color is "B" and value >= 1 and value <= 255:
    pi.set_PWM_dutycycle(BLUE_PIN, value)
  else:
    print("Invalid selection.")
  menu()

def promptRandomColorSingle():
  color = raw_input("[R], [G], or [B]: ")
  if color is "R":
    net_num = random.randint(0,6)
    intensity = setValue (net_num)
    pi.set_PWM_dutycycle(RED_PIN, intensity)
  if color is "G":
    net_num = random.randint(0,6)
    intensity = setValue (net_num)
    pi.set_PWM_dutycycle(GREEN_PIN, intensity)
  if color is "B":
    net_num = random.randint(0,6)
    intensity = setValue (net_num)
    pi.set_PWM_dutycycle(BLUE_PIN, intensity)
    menu()

def promptRandomColorAll():
  net_num1 = random.randint(0,6)
  net_num2 = random.randint(0,6)
  net_num3 = random.randint(0,6)
  intensity1 = setValue(net_num1)
  intensity2 = setValue (net_num2)
  intensity3 = setValue (net_num3)
  pi.set_PWM_dutycycle(RED_PIN, intensity1)
  pi.set_PWM_dutycycle(GREEN_PIN, intensity2)
  pi.set_PWM_dutycycle(BLUE_PIN, intensity3)
  menu()

def promptScanSign():
  raw_input("0 is RED\n1 is GREEN\n2 is BLUE\n3 is MAGENTA\n4 is YELLOW\n5 is CYAN")
  temp()

def set_color_strip(color_enum):
  if color_enum == 0:
    pi.set_PWM_dutycycle(RED_PIN, 255)
  elif color_enum == 1:
    pi.set_PWM_dutycycle(GREEN_PIN, 255)
  elif color_enum == 2:
    pi.set_PWM_dutycycle(BLUE_PIN, 255)
  elif color_enum == 3:
    pi.set_PWM_dutycycle(BLUE_PIN, 255)
    pi.set_PWM_dutycycle(RED_PIN, 255)
  elif color_enum == 4:
    pi.set_PWM_dutycycle(RED_PIN, 255)
    pi.set_PWM_dutycycle(GREEN_PIN, 255)
  lif color_enum == 5:
    pi.set_PWM_dutycycle(BLUE_PIN, 255)
    pi.set_PWM_dutycycle(GREEN_PIN, 255)

def setZero():
  pi.set_PWM_dutycycle(RED_PIN, 0)
  pi.set_PWM_dutycycle(GREEN_PIN, 0)
  pi.set_PWM_dutycycle(BLUE_PIN, 0)

def temp():
  color_enum = int(ser.readline())
  setZero()
  set_color_strip(color_enum)

button = Button(23)
button.when_pressed = temp

try:
  while 1:
    pass
except KeyboardInterrupt:
  turnOffLEDs()
  pi.stop()
