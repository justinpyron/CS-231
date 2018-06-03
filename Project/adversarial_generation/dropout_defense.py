import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import utils


class Dropout_defense:
    '''
    Class that helps experiment with dropout defense strategy
    '''
    def __init__(self, 
                 dataset, 
                 dropout_prob, 
                 file_name=None, 
                 original_label=None, 
                 target_label=None):
        '''
        Arguments:
        - dataset: 'cifar' or 'mnist'
        - dropout_prob: probability of dropping nodes in model forward pass
        - file_name: the name of the file where the adversarial data is
                     stored (this is a .npz file)
        - original_label: original label of images
        - target_label: label of class the model was tricked into predicting
        '''

        if file_name is None:
            if dataset == 'cifar':
                file_name = 'cifar_eps_5e-3'
            if dataset == 'mnist':
                pass
                # file_name = ''    <--------------------------------------------------  Fill this in later

        # Note: we make batch size equal to one
        self.data = utils.get_adv_data(file_name, 
                                       original_label=original_label, 
                                       target_label=target_label, 
                                       batch_size=1)
        self.model = utils.trained_model(dataset)
        self.model.train() # put model in train mode to enable dropout


    def ensemble_forward_pass(self, image, ensemble_size):
        '''
        Perform forward pass multiple times, with dropout enabled, for 
        one single image
        Arguments:
        - image: a pytorch tensor of shape (1,C,H,W), where C = channels,
                 H = height, W = width
        - ensemble_size: the number of forward passes to perform
        '''
        image_set = image.clone().repeat(ensemble_size,1,1,1)
        output = self.model(image_set)



    def visualize(self, input, output):
        '''
        Use this to output plots of ensemble outputs.
        Do this for a random batch of data.
        Arguments:
        - input: the image for which we computed a forward pass
        - outputs: the output from a forward pass. Should be of shape (n,k),
                   where n = ensemble_size, and k = number of classes
        '''



        # Maybe try out seaborn here?
        pass


    def make_dataset(self):
        '''
        Use this to make a dataset that you'll use to train a classifier
        for the "indirect approach"
        '''
        pass

# when performing the ensemble forward pass, make a "batch" containing 
# several copies of the same image. This will make things more 
# computationally efficient.






