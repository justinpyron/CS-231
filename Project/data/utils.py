import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt


class cifar(nn.Module):
    def __init__(self):
        super(cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(40, 40, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(40, 20, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(20*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d( F.relu(self.conv2(self.conv1(x))), 2)
        x = F.max_pool2d( F.relu(self.conv4(self.conv3(x))), 2)
        x = F.max_pool2d( F.relu(self.conv6(self.conv5(x))), 2)

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(20*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d( self.conv2(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d( self.conv4(self.conv3(x)), 2))
        x = x.view(-1, 20*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def trained_model(dataset, path_to_pretrained=None):
    '''
    Return a pre-trained model corresponding to either CIFAR10 or MNIST
    Arguments:
    - dataset: string equal to 'cifar' or 'mnist' specifying the type of model
    - path_to_pretrained: a string indicating path to file where pre-trained 
                          parameter weights are stored
    '''
    assert (dataset == 'cifar' or dataset == 'mnist'), \
                'Model must either be \'cifar\' or \'mnist\''

    if path_to_pretrained is None:
        if dataset == 'cifar':
            path_to_pretrained = 'cifar_model_first_attempt'
        if dataset == 'mnist':
            path_to_pretrained = 'mnist_nn_model'

    if dataset == 'cifar':
        # Initialize model
        model = cifar()
        # Load pre-trained parameters
        model.load_state_dict(torch.load(path_to_pretrained)['state_dict'])
        return model

    if dataset == 'mnist':
        # Initialize model
        model = mnist()
        # Load pre-trained parameters
        model.load_state_dict(torch.load(path_to_pretrained)['state_dict'])
        return model



def get_data(dataset, _train=True, _transforms=transforms.ToTensor(), _batch_size=50):
    '''
    Returns a Pytorch dataloader for the dataset specified by argument dataset
    Arguments:
    - dataset: string equal to 'cifar' or 'mnist' specifying the type of data to load 
    - _train: Boolean indicating whether function should return train or test data
    - _trainsforms: torchvision transformations to apply to dataset
    - _batch_size: integer specifying size of batches in dataset

    NOTE: this function assumes that the CIFAR and MNIST dataset directories
          are in the same directory as this script
    '''

    assert (dataset == 'cifar' or dataset == 'mnist'), \
                'Dataset must either be \'cifar\' or \'mnist\''

    if dataset == 'cifar':
        data = torchvision.datasets.CIFAR10('./', train=_train, transform=_transforms)

    if dataset == 'mnist':
        data = torchvision.datasets.MNIST('./', train=_train, transform=_transforms)

    dataloader = DataLoader(data, batch_size=_batch_size, shuffle=True, num_workers=1)
    return dataloader












