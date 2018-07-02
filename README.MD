# **Behavioral Cloning** 
One center piece in the quest for autonomous cars is called behavioral cloning. By turning human driving in machine readable information it is possible to train a computer to learn to steer a car in the correct direction. While a lot of different approaches have been undertaken to solve this task, a very promosing avenue that has been developed in the last decade is deep learning. Deep neural networks seem to have great capabilities to act on image data. In this project a convolutional neural network (CNN) is created that is capable of driving a vehicle in a computer simulation without human intervention. The final model scores a Mean Squared Error of .0176.

## Introduction
The following project leverages image and steering data collected from a car simulator to create a convolutional neural network that learns to predict the correct steering angle and ultimately is able to drive itself. Code and weights can be found in following files:

- **model.py:** script to create and train the model
- **drive.py:** for driving the car in autonomous mode
- **model.h5:** contains the trained convolutional neural network 
- **evaluate.py:** contains the evaluation code

Using a Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing following command:

```
sh python drive.py model.h5
```

![alt text][image1]

## Model Architecture and Training Strategy
The final model draws inspiration from the NVIDIA paper [End-to-End Deep Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). It contains four convolutional layers with a filter size of 5x5 and 3x3 as well as increasing depths from 24 to 64. Afterwards the information is funneled through four dense layers with decreasing size. While the original architecture works with fewer and smaller dense layers, the additional layers seem to have a smoothing effect on the predicted values. After implementing the additional depth of the network the car steers less 'choppy'. A cropping layer removes some of the unnecessary information by reducing the image size to 90x320. Afterwards normalization is applied to help the model converge quicker. An overview of the architecture can be seen below.


![alt text][image2]

## Report and Video
A full report on training as well as data selection is available in writeup_report.pdf. A video of the final model running on auto-pilot can be seen [here](https://www.youtube.com/watch?v=aBFZFNgPO5E).

[//]: # (Image References)

[image1]: ./images/sample_simulator.png "Sample Simulator Image"
[image2]: ./images/neural_net.png "CNN Architecture"