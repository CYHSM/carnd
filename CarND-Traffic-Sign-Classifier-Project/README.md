# Project Two
The second project of the udacity self-driving car course challenged us to classify traffic signs, taken from a dataset of german traffic signs. Click [here](https://github.com/CYHSM/carnd/blob/master/CarND-Traffic-Sign-Classifier-Project/README_Udacity.md) for the official Udacity readme regarding this project.

Solving this project required training of a deep neural network. I outline the steps I used below. I achieved an Validation accuracy of 0.995 and a Test accuracy of 0.951.

## The Dataset
The traffic dataset comes from the neuroinformatics institute of the Ruhr-University Bochum ([GTS](benchmark.ini.trub.derub.de)). It consists of 51839 pictures of traffic signs divided into 43 classes. Here is an overview over the different classes and their corresponding number of images.
![sample_dataset](https://github.com/CYHSM/carnd/blob/master/CarND-Traffic-Sign-Classifier-Project/sample_dataset.png)
As can be already seen there are some images which are very hard to classify for a number of different reasons:
1. the size of the images is just 32x32 pixels
2. huge brightness levels between very dark and very bright
3. blurry images (caused by the extraction from a video)

However one of the most important parts of detecting traffic signs in images is already taken care of in this dataset, namely the detection of the traffic sign in the image. All of these are already cropped in order center the traffic sign in the image.

## Preprocessing
I tried different preprocessing steps (see [Jupyter Notebook](https://github.com/CYHSM/carnd/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb) for a full list) but in the end settled for:

1. Scale data to (0,1)
2. Convert images to grayscale

as these two gave the best accuracy improvement.

## Class Balancing
I balanced the classes in order to overcome the limitations during classification of under-sampled classes. In order to achieve a uniform class distribution, under-sampled classes were filled with augmented images were I used either a left or right perspective transformation.

Right Perspective Transformation             |  Left Perspective Transformation
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-Traffic-Sign-Classifier-Project/perspective_right.png?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-Traffic-Sign-Classifier-Project/perspective_left.png?raw=true)

## Network Architecture
I used a modified version of the LeNet architecture with following layers:
* Layer 1 : 5x5 Filter with depth 12
* Layer 2 : 5x5 Filter with depth 32
* Fully Connected Layer : n = 512
* Dropout Layer : Dropout Value = 0.8
* Fully Connected Layer : n = 256
* Dropout Layer : Dropout Value = 0.8

see [Jupyter Notebook](https://github.com/CYHSM/carnd/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb) for more details on the architecture.
