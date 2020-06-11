##############################################################################################################################
# Importing necessary libraries
##############################################################################################################################

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np

##############################################################################################################################
# Creating parser for arguments in command line
##############################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('data_dir',
                    type = str,
                    default = 'flowers',
                    help = 'Directory for folder with data for image classifier to train and test')
parser.add_argument('--save_dir',
                    action = 'store',
                    type = str,
                    default = 'checkpoint.pth',
                    help = 'Name of file for trained model')
parser.add_argument('--arch',
                    action = 'store',
                    type = str,
                    default = 'vgg11',
                    choices = ['vgg11', 'vgg13', 'vgg19'],
                    help = 'vgg11, vgg13, vgg19')
parser.add_argument('--learning_rate',
                    action = 'store',
                    type = float,
                    default = 0.001,
                    help = 'A float number as the learning rate for the model')
parser.add_argument('--hidden_units',
                    action = 'store',
                    type = int,
                    default = 4096,
                    help = 'Number of hidden units for 1st layer')
parser.add_argument('--epochs',
                    action = 'store',
                    type = int,
                    default = 5,
                    help = 'Number of epochs for gradient descent')
parser.add_argument('--gpu',
                    action = 'store_true',
                    default = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    help = 'GPU if available')

args = parser.parse_args()

if args.data_dir:
    data_dir = args.data_dir
if args.save_dir:
    save_dir = args.save_dir
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:
    device = args.gpu

##############################################################################################################################
# Loading training, validation, and testing data
##############################################################################################################################

def dataloaders(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=17, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=17)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=17)

    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data

train_loader, valid_loader, test_loader, train_data, valid_data, test_data = dataloaders(data_dir)
print('Data has been loaded...')

##############################################################################################################################
# Defining a classifier with a pretrained network
##############################################################################################################################

def network(arch = arch , hidden_units = hidden_units , learning_rate = learning_rate):

    model =  getattr(models,arch)(pretrained=True)
    input = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
                              nn.Linear(input, hidden_units),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(hidden_units, 512),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(512, len(train_data.class_to_idx)),
                              nn.LogSoftmax(dim=1)
                              )

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

    return model, classifier, criterion, optimizer

model, classifier, criterion, optimizer = network()
print('Model has been built...')
device = args.gpu
model.to(device);

##############################################################################################################################
# Training the pretrained network with defined classifier
##############################################################################################################################
def trained_network(model, train_loader, valid_loader, epochs = epochs , learning_rate = learning_rate):

    print('Training model...')
    model.to(device);
    steps = 0
    train_losses, valid_losses = [], []

    for i in range(epochs):
        running_loss = 0

        for images,labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():

                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    valid_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()
            train_losses.append(running_loss/len(train_loader))
            valid_losses.append(valid_loss/len(valid_loader))

            print("Epoch: {}/{}.. ".format(i+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

    return model

model = trained_network(model, train_loader, valid_loader)
print('Model has been trained...')

##############################################################################################################################
# Determining the test accuracy
##############################################################################################################################
def test_accuracy(model):

    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Testing Accuracy: {:.3f}".format(accuracy/len(test_loader)))

    return

test_accuracy(model)


##############################################################################################################################
# Saving a checkpoint of the trained model
##############################################################################################################################

def save_checkpoint(model):

    model.class_to_idx = train_data.class_to_idx
    model.name = arch
    model.epochs = epochs
    model.learning_rate = learning_rate
    model.hidden_units = hidden_units
    model.output = len(train_data.class_to_idx)

    checkpoint = {'classifier': classifier,
                  'arch': model.name,
                  'epochs': model.epochs,
                  'learning_rate': model.learning_rate,
                  'hidden_units': model.hidden_units,
                  'output': len(train_data.class_to_idx),
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, save_dir)

    return

save_checkpoint(model)

print('Model has been saved!')


##############################################################################################################################
# Running file instructions
##############################################################################################################################

#To run enter on command line:
#cd ImageClassifier
#python train.py flowers
