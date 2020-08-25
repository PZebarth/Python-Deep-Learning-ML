# Developing an AI application
## Project Motivation
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

## Project Description
In this project, I'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories. 

## Project Process
The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on our dataset
* Use the trained classifier to predict image content

## File Description
Although the data is too large to have here, we will use [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories with the following files: <br> 

>Image Classifier Project.ipynb: detailed analysis of project
>train.py: command line application which uses pretrained network from DenseNet
>predict.py: predicts the flower classification based on image file


