##############################################################################################################################
# Importing necessary libraries
##############################################################################################################################

import argparse
import torch
import json
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

##############################################################################################################################
# Creating parser for arguments in command line
##############################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('image_path',
                    type = str,
                    default = 'flowers/test/1/image_06743.jpg',
                    help = 'path to image file for model classification of top k, flowers/test/class/image')
parser.add_argument('checkpoint',
                    action = 'store',
                    type = str,
                    default = 'checkpoint.pth',
                    help = 'Name of file for trained model')
parser.add_argument('--top_k',
                    action = 'store',
                    type = int,
                    default = 3,
                    help = 'Select number of classes you wish to see in descending order.')
parser.add_argument('--category_names',
                    action = 'store',
                    type = str,
                    default = 'cat_to_name.json',
                    help = 'Name of json file for class to flower names')
parser.add_argument('--arch',
                    action = 'store',
                    type = str,
                    default = 'vgg11',
                    help = 'vgg11, vgg13, vgg19')
parser.add_argument('--gpu',
                    action = 'store_true',
                    default = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    help = 'GPU if available')

args = parser.parse_args()

if args.image_path:
    image_path = args.image_path
if args.checkpoint:
    checkpoint = args.checkpoint
if args.top_k:
    top_k = args.top_k
if args.category_names:
    category_names = args.category_names
if args.arch:
    arch = args.arch
if args.gpu:
    device = args.gpu

##############################################################################################################################
# Loading saved checkpoint of model
##############################################################################################################################

def load_checkpoint(checkpoint):

     checkpoint = torch.load(checkpoint)
     arch = checkpoint['arch']
     model =  getattr(models,arch)(pretrained=True)

     for param in model.parameters():
         param.requires_grad = False

     model.class_to_idx = checkpoint['class_to_idx']
     input = model.classifier[0].in_features
     hidden_units = checkpoint['hidden_units']
     output = checkpoint['output']

     classifier = nn.Sequential(
                               nn.Linear(input, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, 512),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(512, output),
                               nn.LogSoftmax(dim=1)
                               )

     model.classifier = classifier
     model.load_state_dict(checkpoint['state_dict'])

     return model

model = load_checkpoint(checkpoint)
print('Checkpoint has been loaded...')
print(model)

##############################################################################################################################
# processing image for image_path
##############################################################################################################################

def process_image(image_path):

    im = Image.open(image_path)

    size = 256, 256
    im.thumbnail(size)

    crop_size = 224
    left = (size[0] - crop_size)/2
    upper = (size[1] - crop_size)/2
    right = left + crop_size
    lower = upper + crop_size
    im_crop = im.crop(box = (left, upper, right, lower))

    np_image = (np.array(im_crop))/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean) / std

    processed_image = np_image.transpose(2,0,1)

    return processed_image

process_image(image_path)

##############################################################################################################################
# predicting top_k from processed image
##############################################################################################################################

def predict(image_path, model, top_k):

    model.cpu()
    model.eval()

    image = process_image(image_path)
    tensor = torch.tensor(image).float().unsqueeze_(0)

    with torch.no_grad():
        log_ps = model.forward(tensor)

    ps = torch.exp(log_ps)
    probs, classes = ps.topk(top_k, dim=1)

    return probs , classes

probs, classes = predict(image_path, model, top_k)

##############################################################################################################################
# Predicting top_k from processed image
##############################################################################################################################

def image_prediction(image_path):

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    image = process_image(image_path)

    probs, classes = predict(image_path, model, top_k)
    probs = probs.data.numpy().squeeze()
    classes = classes.data.numpy().squeeze()
    np.set_printoptions(suppress=True)

    idx = {val: i for i, val in model.class_to_idx.items()}
    labels = [idx[labels] for labels in classes]
    flowers = [cat_to_name[labels] for labels in labels]

    print('The predictions for ', image_path, 'are...')
    print(flowers, probs)

    return flowers, probs

image_prediction(image_path)

##############################################################################################################################
# Running file instructions
##############################################################################################################################


#To run enter on command line:
#cd ImageClassifier
#python predict.py flowers/test/1/image_06743.jpg checkpoint.pth
