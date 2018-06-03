import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import utils
from matplotlib import gridspec


class Dropout_defense:
    '''
    Class that helps experiment with dropout defense strategy
    '''
    def __init__(self, 
                 dataset, 
                 file_name=None):
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
                file_name = 'mnist_eps_5e-3_norm_2_num_iters_50.npz'
        self.dataset = dataset
        self.file_name = file_name

        self.model = utils.trained_model(self.dataset)
        self.model.train() # put model in train mode to enable dropout

    def reset_model(self, dropout_prob):
        '''
        Initialize a new model to use with object's data. This is mainly used
        to reset the dropout probability parameter.
        '''
        self.model = utils.trained_model(self.dataset, dropout_prob)
        self.model.train()


    def ensemble_forward_pass(self, image, ensemble_size):
        '''
        Perform forward pass multiple times, with dropout enabled, for 
        one single image
        Arguments:
        - image: a pytorch tensor containing data for one single example. 
                 Should be of shape (1,C,H,W), where C = channels, 
                 H = height, W = width
        - ensemble_size: the number of forward passes to perform
        '''
        image_set = image.clone().repeat(ensemble_size,1,1,1)
        output = self.model(image_set)
        return output


    def plot_image(self, 
                   original, adversary, 
                   original_output, adv_output, 
                   original_label, target_label):
        '''
        Plots ensemble outputs for a single image juxtaposed with the original
        and adversarial image.
        Arguments:
        - original: torch tensor containing pixel data for original image
        - adversary: torch tensor containing pixel data for adversarial image
        - outputs: the output from a forward pass. Should be of shape (n,k),
                   where n = ensemble_size, and k = number of classes
        '''

        gs = gridspec.GridSpec(1,4, width_ratios=[1,1,4,4], wspace=0.6)
        plt.figure(figsize=(16, 4))

        plt.subplot(gs[0])
        if self.dataset == 'cifar':
            plt.imshow(original.squeeze(0).numpy().transpose(1,2,0))
        if self.dataset == 'mnist':
            plt.imshow(original.squeeze(0).squeeze(0).numpy(), cmap='binary_r')
        plt.title('Original Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(gs[1])
        if self.dataset == 'cifar':
            plt.imshow(adversary.squeeze(0).numpy().transpose(1,2,0))
        if self.dataset == 'mnist':
            plt.imshow(adversary.squeeze(0).squeeze(0).numpy(), cmap='binary_r')
        plt.title('Adversarial Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(gs[2])
        colors = ['lightgrey']*10
        colors[original_label] = '#1f77b4' # blue
        plt.bar(range(10), original_output/original_output.sum(), color=colors)
        plt.xticks(range(10))
        plt.xlabel('Predicted Class')
        plt.ylabel('Fraction of Predictions')
        plt.title('Original Image')
        plt.grid(alpha=0.25)

        plt.subplot(gs[3])
        colors = ['lightgrey']*10
        colors[original_label] = '#1f77b4' # blue
        colors[target_label] = '#ff7f0e' # orange
        plt.bar(range(10), adv_output/adv_output.sum(), color=colors)
        plt.xticks(range(10))
        plt.xlabel('Predicted Class')
        plt.ylabel('Fraction of Predictions')
        plt.title('Adversarial Image')
        plt.grid(alpha=0.25)

        plt.show()


    def visualize(self, 
                  dropout_prob, 
                  original_label, 
                  target_label,
                  ensemble_size, 
                  num_to_plot=25):
        '''
        Plots ensemble outputs for a random batch of data.
        Arguments:
        - input: the image for which we computed a forward pass
        - outputs: the output from a forward pass. Should be of shape (n,k),
                   where n = ensemble_size, and k = number of classes
        '''
        # Get data
        # Note: we make batch size equal to one
        data = utils.get_adv_data(self.file_name, 
                                   original_label=original_label, 
                                   target_label=target_label, 
                                   batch_size=1)

        # Reset model to have proper dropout probability
        self.reset_model(dropout_prob=dropout_prob)

        for (i, data_tuple) in enumerate(data):
            if i >= num_to_plot:
                break
            original, pert, adv, orig_label, target_label = data_tuple # unpack
            original_ensemble = torch.argmax(self.ensemble_forward_pass(
                                             original, ensemble_size), dim=1)
            adv_ensemble = torch.argmax(self.ensemble_forward_pass(
                                        adv, ensemble_size), dim=1)

            original_counts = np.eye(10)[original_ensemble.numpy()].sum(axis=0)
            adv_counts = np.eye(10)[adv_ensemble.numpy()].sum(axis=0)

            self.plot_image(original, adv, 
                            original_counts, adv_counts, 
                            original_label, target_label)


    def get_data_subset(self, original_label, target_label):
        '''
        Returns a subset of the data where the original label was 
        original_label and the target label was target_label
        Arguments
        - original_label: the original label of images
        - target_label: the label that adversaries trick model into predicting
        '''
        # Maybe do this later. It would prevent you having to read in the 
        # entire dataset and then subset it each time. But this isn't too
        # big of a deal
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






