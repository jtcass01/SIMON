# SIMON
**SIMON** - Sign Interfaced Machine Operating Network

Neural Network Models Implemented by Jacob Cassady

**1.1 Team Members** - Jacob Cassady, Aaron Fox, Nolan Holdan, Kristi Kidd, Jacob Marcum
- C and Assembly code developed by Jacob Marcum.

**1.2 Team Members** - Jacob Cassady, Patrick Chuong, Mitchel Johnson, Oscar Leon, Matthew Long
- Sections of python code controling LED lights developed by Patrick Chuong and Matthew Long.

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
The structure of SIMONs ResNet50 model was implemented using notes taken from Professor Andrew Ng’s deeplearning.ai specialization on coursera. The figures displayed in this section were taken from his notes. A generated graph of SIMON’s implemented ResNet50 can be found [here](https://github.com/jtcass01/SIMON/blob/master/1.2/models/best_model.png).

<img src="1.2/reference_images/resnet50_model.png" style="width:450px;height:220px;">

**Figure 1**  : **ResNet50 Model** <br> The above figure displays at a high level the structure of the ResNet50 model. It contains a series of convolutional blocks followed by identity blocks with a head and a tail for initial padding and final prediction. Below is a table tracking the image dimensions as it propagates through the network.

<table> 
    <tr>
        <td> 
          Stage
        </td>
        <td> 
          Image Dimension at Stage Output
        </td> 
    </tr>
    <tr>
        <td> 
          Input
        </td>
        <td> 
          (64,64,3)
        </td> 
    </tr> 
    <tr>
        <td> 
           Stage 1
        </td>
        <td> 
           (64,64,128)
        </td> 
    </tr>
    <tr>
        <td> 
           Stage 2
        </td>
        <td> 
           (64,64,256)
        </td> 
    </tr>
    <tr>
        <td> 
           Stage 3
        </td>
        <td> 
           (128,128,256)
        </td> 
    </tr>
    <tr>
        <td> 
           Stage 4
        </td>
        <td> 
           (256,256,1024)
        </td> 
    </tr>
    <tr>
        <td> 
           Stage 5
        </td>
        <td> 
           (512,512,2048)
        </td> 
    </tr>
    <tr>
        <td> 
           Output Layer
        </td>
        <td> 
           (1,6)
        </td> 
    </tr>
</table>

**Table 1**  : **ResNet50 Image Dimensions**

<img src="1.2/reference_images/identity_block.png" style="width:450px;height:220px;">

**Figure 2**  : **Identity Block** <br> As mentioned previously, the identity block is important for allowing the network to learn the identity function. As you can see in the figure, the input from the start of the neural chain is summed with the output of a later block before the block enters its activation function. It is important to note the dimensions of the input and later output must be the same for the addition to work. Therefore, the dimensions of the input and output of each identity block can be assumed to be the same.

<img src="1.2/reference_images/convolutional_block.png" style="width:450px;height:220px;">

**Figure 3**  : **Convolutional Block** <br> The Convolutional block is like the identity block but includes an additional 2D convolutional layer during the skip connection. This allows for the changing of dimension during the convolutional block while still gaining some of the benefits from the increased complexity of the skip connection.

##### 1.1.2 Implementation
Please see ResNet50.py in the Neural Network section of Source Code within this document.

The ResNet50 model was implemented in keras, a high-level neural network framework with tensorflow underpinnings. Keras was chosen because it greatly reduces the time, thought, and lines of code required to implement complex deep neural networks. This is because you don’t have to implement the backpropagation portions and can utilize predefined loss and optimizer functions. SIMON’s ResNet50 model was compiled with the adam optimizer and categorical cross entropy loss function.

Models were logged and loaded using HDF5 binary compression and the h5py library. JSON and graphical descriptions of the models were also generated when each model was saved and are stored alongside the saved models.

##### 1.1.2 Results
A training framework was built within SIMON for training a series of models and choosing the best one. 1200 images of 6 different ASL representations (200 images per representation) were used for training and testing the models with 1080 images in the training set and 120 images in the test set. Models were trained for 20 epochs using a minibatch size of 32. The models were then compared using the test set to determine minimal variance.

After a few days of training, the best ResNet50 model correctly categorized all the training set images and 114 of 120 (95%) of the test set images. This shows a working model that has low bias and variance. It is important to note the training and test set images come from the exact same data distribution.

SIMON requests the user to choose an image from their directory for prediction. Images during deployment of SIMON were taken using a laptop’s built-in 1080-720p camera. These images were then scaled using the scipy module to 64-64p images. When attempting to use these images from a different and distorted distribution, the ResNet50 model always produced a prediction of 0 but would correctly predict all individually tested images from the same data distribution. Consequently, the ResNet50 model is not a good candidate for controlling a device until the train and test set include a more representative distribution to that used in deployment.

### 1.2 Classic Deep Neural Network Model

## References
 Microchip Technology Inc. (2018, January 3). AVR Assembler Instructions. Retrieved January
         30, 2018, from http://www.microchip.com/webdoc/avrassembler/avrassembler.wb_instru 
         ction_list.html ATmega328P Xplained Mini[PDF]. (2017). Chandler, AZ: Microchip 
         Technology, Inc

Atmel Corporation. (2016). ATmega328/P Datasheet Complete [PDF]. Atmel.

Specifications for Liquid Crystal Display[PDF]. (n.d.). AZ Displays, Inc.

ATmega328P Xplained Mini[PDF]. (2017). Chandler, AZ: Microchip Technology, Inc.

Notes for Deep Neural Network were taken from Stanford Professor Andrew Ng’s Deep Learning Specialization on coursera.org. https://www.coursera.org/specializations/deep-learning



