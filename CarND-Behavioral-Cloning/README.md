# Project Three
In this project we had to develop a model which should be able to stear a car autonomously around a simulated track. The simulator had two modes, one where we could obtain training data and one for testing our model.

## Training Data
Obtaining 'good' training data was one of the most tricky parts of this projects. I recorded the car on 5 consecutive laps where I tried to stear the car in the center of the road. 2 more Laps were added in which the curves were driven in a very narrow angle and additional recovery data was also recorded. Recovery was performed by stearing towards the edges and before the car could leave the track it was steared back towards the center of the road.

### Camera Angles
Each frame consisted of three images (left, center, right):

Left | Center | Right
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-Behavioral-Cloning/Samples/left.jpg?raw=true)|![](https://github.com/CYHSM/carnd/blob/master/CarND-Behavioral-Cloning/Samples/center.jpg?raw=true)|![](https://github.com/CYHSM/carnd/blob/master/CarND-Behavioral-Cloning/Samples/right.jpg?raw=true)

In my model all three images are used, where the left and the right images have an offset added to their steering angles. I experimented a bit with values between 0.1-0.5 and ended up using +0.25 for the left image and -0.25 for the right image.

## Model Architecture
My model architecture consists of convolutional layers followed by fully connected layers with added dropout:
![](https://github.com/CYHSM/carnd/blob/master/CarND-Behavioral-Cloning/model_architecture.png?raw=true)

## Model Evaluation

### Generator
I used an generator for training the network which yields batches infinitely. This helps with memory restrictions as the training data was quite big and would not have fitted into one variable!

### Metric
As this project consisted of a regression accuracy metrics would not have yielded good results. Also the MSE is not the best indicator of future performance on the track. I saw this when evaluating the car with an offset of 0.4 as it was super jumpy but was able to take some curves, while the training showed a worse MSE than for lower offset values. Therefore I only used the autonomous simulator as an validation approach.

### Optimization
I ended up using the adam optimizer with the default learning rate. I tried to tune it but the resulting behavior on the track did not show any improvements over the 0.001 lerning rate or was worse!
