# SIMON
**SIMON** - Sign Interfaced Machine Operating Network

Neural Network Models Implemented by Jacob Cassady

**1.1 Team Members** - Jacob Cassady, Aaron Fox, Nolan Holdan, Kristi Kidd, Jacob Marcum

C and Assembly code developed by Jacob Marcum.

**1.2 Team Members** - Jacob Cassady, Patrick Chuong, Mitchel Johnson, Oscar Leon, Matthew Long




## Abstract
The purpose of the Sign-Interfaced Machine Operating Network, or SIMON, is to develop a machine learning classifier capable of detecting a discrete set of American Sign Language (ASL) presentations from captured images of a hand and producing a desired response on another system.  SIMON can utilize a variety of neural networks (nets) for producing its predictions including a classic deep neural net and a ResNet50 convolutional model.  SIMON’s neural nets were trained on 6 targets using 1200 images (200 of each representation).  The training and test sets included 1080 and 120 images respectively.  SIMON utilizes its neural nets to produce a discrete value from an image input matching the target’s class enumeration.  Images for demonstration were taken using a laptop webcam.  Predictions are represented in ASCII and are sent across a serial connection to a response machine.  This project created a response machine from a Raspberry Pi Model 3 B+, a 5050 LED light strip, and a button.   The goal of this project was to produce the correct color response from the LED light strips given an image containing an ASL representation included in our models’ training.  This goal was accomplished at its most baseline level although there were some issues and inconsistencies in the neural nets performances including differences in data distributions from training to production as well as distortions in the production data due to preprocessing techniques. SIMON was developed using Python 3.6 as well as a variety of 3rd party modules and tested on both Windows 10 and Ubuntu 17.

## Body
The purpose of Sign-Interfaced Machine Operating Network, or SIMON, is to develop a machine learning classifier capable of detecting a discrete set of American Sign Language (ASL) presentations from captured images of a hand and producing a desired response on another system.  The response system (LED Strip) was developed on a Raspberry Pi 3 B+ utilizing the Raspbian OS and Python 3.6, along with one 5050 LED light strip, and one button.

### 1. Sign-Interfaced Machine Operating Network (SIMON)
SIMON was developed in Python 3.6 and tested on both Windows 10 and Ubuntu 17 operating systems.  SIMON has access to two neural network models: the convolutional ResNet50 and a classic fully connected 3-layer deep neural network model.  The neural nets were implemented using the keras and tensorflow neural net libraries.  Additional 3rd party libraries were utilized for data processing and matrix manipulation including NumPy, scipy, and h5py. Furthermore, tkinter was leveraged for a graphical display during file selection and the pyserial library was utilized for communication with the response system.

#### 1.1	ResNet50 Model
Convolutional neural networks utilize a mathematical formula similar to a convolution as well as padding and filters to shape an image while it progresses through a neural net.  The use of these convolution-like mathematical operators allow for the training of a greater number of parameters while using less computational power and time than standard forward propagation over a one-dimensional matrix representing an equal number of pixels.  This is because convolutional models have fewer connections between neurons and allow for the sharing of parameters.  The ResNet50 is a convolutional deep neural network model.  It includes 152 layers of neurons as well as over 23.5 million trainable parameters.

Previously it was thought that very deep neural networks were impossible to train due to exploding and vanishing gradients (Pascanu, Mikolov, & Bengio, 2012).  The basic concept driving Residual Networks (He, Xiangyu, Shaoqing, & Jian, 2015) are “skip connections” which hope to solve the issue of exploding or vanishing gradients.  A “skip connection” is when you utilize part or all of the activation of one layer to alter the input to a subsequent layer deeper in the neural network.  The idea is that these skip connections give sections of the neural net the ability to learn the identity function, allowing the neural net to utilize the maximum number of needed neural layers while making the addition of more layers undamaging to its optimal accuracy.  Additionally, it is theorized this ability to learn the identity function yields less of a chance of causing an exploding or vanishing gradient.

##### 1.1.1 Structure
..to be continued.

## References
 Microchip Technology Inc. (2018, January 3). AVR Assembler Instructions. Retrieved January
         30, 2018, from http://www.microchip.com/webdoc/avrassembler/avrassembler.wb_instru 
         ction_list.html ATmega328P Xplained Mini[PDF]. (2017). Chandler, AZ: Microchip 
         Technology, Inc

Atmel Corporation. (2016). ATmega328/P Datasheet Complete [PDF]. Atmel.

Specifications for Liquid Crystal Display[PDF]. (n.d.). AZ Displays, Inc.

ATmega328P Xplained Mini[PDF]. (2017). Chandler, AZ: Microchip Technology, Inc.

Notes for Deep Neural Network were taken from Stanford Professor Andrew Ng’s Deep Learning Specialization on coursera.org. https://www.coursera.org/specializations/deep-learning



