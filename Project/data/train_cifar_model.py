
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
import matplotlib.pyplot as plt
import time


class cifar(nn.Module):
    def __init__(self):
        super(cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(100, 100, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(100, 150, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(150, 150, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(150*4*4, 10)

    def forward(self, x):
        x = F.max_pool2d( F.relu(self.conv2(self.conv1(x))), 2)
        x = F.max_pool2d( F.relu(self.conv4(self.conv3(x))), 2)
        x = F.max_pool2d( F.relu(self.conv6(self.conv5(x))), 2)
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)
        return x



# Functionality to save model parameters
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def train(epoch, saved_model_path, log_interval=100):
    start = time.time()
    model.train()  # set training mode
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
                print('Total time elapsed (in minutes) {:.3f}'.format( (time.time() - start)/60. ))

            iteration += 1
        test(saved_model_path)
    save_checkpoint('{}/final_model'.format(saved_model_path), model, optimizer)
    

def test(saved_model_path):
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0
    best_accuracy = 0.0
    with torch.no_grad():
        for data, target in testset_loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    
    if correct/len(testset_loader.dataset) > best_accuracy:
        save_checkpoint('{}/cifar_resnet_best_model'.format(saved_model_path), model, optimizer)
        


if __name__ == '__main__':

    cifar_train = torchvision.datasets.CIFAR10('./',
                                               train=True,
                                               transform=transforms.ToTensor())

    cifar_test = torchvision.datasets.CIFAR10('./',
                                              train=False,
                                              transform=transforms.ToTensor())

    trainset_loader = DataLoader(cifar_train, batch_size=50, shuffle=True, num_workers=1)
    testset_loader = DataLoader(cifar_test, batch_size=50, shuffle=True, num_workers=1)

    model = cifar()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)

    num_epochs = 1
    # Train model and save trained parameters
    train(num_epochs, 'cifar_saved_models')



