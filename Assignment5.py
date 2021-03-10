# Note, data for this program should live in /data/celeba
# I have that in gitignore b/c it's too large to hold in github. This program won't work unless you download
# The needed dataset and point data root to it.

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from time import time

data_root = "data/celeba"
num_epochs = 5
lr = 0.0002  # Learning Rate
workers = 2
batch_size = 128
image_size = 6  # Spatial size of training images. All images will be resized to this size in transform.
nc = 3  # Number of color channels
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
beta1 = 0.5  # Beta1 hyper parameter for Adam optimizers
ngpu = 1  # Num GPUs available -- I have 1.


if __name__ == "__main__":
    print("Start GAN")

    start_time = time()
    time_to_fit = time() - start_time
