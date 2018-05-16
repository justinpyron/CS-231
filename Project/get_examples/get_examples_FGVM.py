
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


class MNIST_specific_label(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self,
                 root,
                 class_label,
                 transform=None):
        """ Create a dataset of images from MNIST belonging to a specific class
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - class_label: the label of the class to generate dataset for
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        filenames = glob.glob(osp.join(root, str(class_label), '*.png'))

#         for fn in filenames:
        for fn in filenames[:10]:    # <------------------------------------------------------------------------------  CHANGE THIS
            self.filenames.append((fn, class_label)) # (filename, label) pair
                
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


#----------------------------------------------------------
# Define a model architecture
# This is the model architecture that our adversarial
# images trick. A model with this architecture was trained 
# for 10 epochs on the entire MNIST dataset and achieved
# a final test set accuracy of 98%.
#----------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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



def save_image(tensor, path):
    '''
    Save a PyTorch Tensor as a PNG file
    Arguments:
        - tensor: a PyTorch tensor. Expected dimensions are (C,H,W),
                    where C = channels, H = height, W = width
        - path: where to save file
    '''
    # Make tensor values fall in range [0,1]
    tensor = tensor - tensor.min()
    tensor = tensor/tensor.max()
    tensor = transforms.ToPILImage()(tensor)
    tensor.save(path)



def batch_FGVM(input_imgs, model, target, epsilon=5e-2, num_iters=1000):
    '''
    Generate adversarial images utilizing the Fast Gradient Value method
    As described on page 6 here: https://arxiv.org/abs/1712.07107
    Arguments:
        - input_imgs: torch tensor of input images
        - model: model used to make predictions
        - target: class to trick the model into predicting
        - epsilon: scalar by which to multiply gradient when perturbing natural input
        - num_iters: maximum number of times to add a perturbation to image
    '''
    batch_size = input_imgs.size()[0]
    input_imgs = input_imgs.clone()
    input_imgs.requires_grad_(True)  # very important!
    
    perturbation = torch.zeros_like(input_imgs)
    fooled = False
    iteration = 0

    dout = torch.zeros_like(model(input_imgs), dtype=torch.float)
    dout[:,target] = 1.  # only compute gradient w.r.t. target class
    
    required_iters = torch.zeros(batch_size)

    while fooled is False and iteration < num_iters:
        
        output = model(input_imgs)
        model.zero_grad() # zero out all gradients in model so they don't accumulate
        grad = torch.autograd.grad(outputs=output, inputs=input_imgs, grad_outputs=dout)[0]
        
        with torch.no_grad():
            perturbation.add_(epsilon * grad)
            input_imgs.add_(epsilon * grad)
            
            predictions = torch.argmax(model(input_imgs), 1)
            
            # If an example is correctly predicted, set all upward gradients
            # flowing to that example to zero; we've successfully found an
            # adversarial image that tricks the model and no longer need to 
            # update the original. We keep looping to find successfull
            # adversarial images for the other examples.
            dout[:,target] = (predictions != target)
            required_iters.add_( (predictions != target).type(torch.float) )
            
            # If every adversarial example fooled the model, we're done
            if (predictions == target).sum() == batch_size:
                fooled = True
            iteration += 1
            
    return (perturbation.detach(), input_imgs.detach())




def create_dataset_FGVM(source_path, batch_size=100):
    '''
    Create dataset of adversarial MNIST images
    arguments:
        - source_path: path of directory holding original MNIST images
    '''

    start = time.time()

    # Set device
    torch.manual_seed(123)
    device = torch.device('cpu')

    # Initialize a model and load pretrained parameters
    model = Net().to(device)
    state = torch.load('mnist_nn_model')
    model.load_state_dict(state['state_dict'])
    
    # Create directory where dataset will be stored
    parent_dir = 'MNIST_adversarial_FGVM'
    os.makedirs(parent_dir)
    
    for class_label in range(10):
        print('\n\nWorking on class label {}'.format(class_label))
        
        # Create a dataset containing images only of a specific class
        dataset = MNIST_specific_label(root=source_path,
                                       transform=transforms.ToTensor(),
                                       class_label=class_label)
        dataset_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1)

        # Create directory to store images belonging to class_label
        child_dir = '{}{}{}'.format(parent_dir, '/original_', class_label)
        os.makedirs(child_dir)
        
        image_number = 0 # use this when naming files

        # Loop through batches of dataset
        for batch_num, (batch_data, batch_label) in enumerate(dataset_loader):
            if batch_num % 10 == 0:
                print('Class label: {},  Batch {}/{}'.format(class_label, batch_num, len(dataset_loader)))
            
            # Only keep images that model correctly predicts
            predictions = model(batch_data).argmax(1) # max along axis=1
            batch_data = batch_data[predictions == class_label]
            
            # Store tensors containing batches of adversarial images in a dict
            adv_dict = dict()
            
            for adversarial_label in range(10):
                if adversarial_label == class_label:
                    continue
                
                # Generate adversarial images
                perturbations, adv_images = batch_FGVM(input_imgs=batch_data, 
                                                      model=model, 
                                                      target=adversarial_label)
                adv_dict[adversarial_label] = adv_images
            
            # Save original and adversarial images
            for i in range(len(batch_data)):
                # Make a folder corresponding to each image
                grandchild_dir = '{}{}{}'.format(child_dir, '/image', image_number)
                os.makedirs(grandchild_dir)

                # Save original image
                file_name = '{}{}'.format(grandchild_dir, '/original.png')
                save_image(batch_data[i], file_name)
                
                # Save adversarial images
                for label, adv_images in adv_dict.items():
                    file_name = '{}{}{}{}'.format(grandchild_dir, '/target_', label, '.png')
                    save_image(adv_images[i], file_name)
                
                image_number += 1
    print('\nFinished making adversarial dataset!\nTook {0:.2f} minutes\n'.format( (time.time() - start)/60. ))


if __name__ == '__main__':
  # Note: only using MNIST training data to greate adversarial images
  create_dataset_FGVM(source_path='mnist_png/training')

