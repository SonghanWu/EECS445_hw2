"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Target(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        # define convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)  # SAME padding
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)  # SAME padding
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, stride=2, padding=2)  # SAME padding
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)  # 2(f) iii.

        # define fc
        self.fc_1 = nn.Linear(in_features=8 * 2 * 2, out_features=2)  # Output classes are 2
        # self.fc_1 = nn.Linear(in_features=64 * 2 * 2, out_features=2)  # 2(f) iii.

        # initialize weight
        self.init_weights()


    def init_weights(self):
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            ## TODO: initialize the parameters for the convolutional layers
            # Followed the instruction in Appendix B: "Weight initialization: normally distributed with ... Bias initialization: constant ..."
            nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1 / (5 * 5 * conv.in_channels)))
            nn.init.constant_(conv.bias, 0.0)
        
        ## TODO: initialize the parameters for [self.fc1]
        # initialize the weight and bias of the fully connected layer
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=sqrt(1 / self.fc_1.in_features))
        nn.init.constant_(self.fc_1.bias, 0.0)


    def forward(self, x):
        N, C, H, W = x.shape

        # Convolutional Layer 1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (16, 32, 32)
        
        # Convolutional Layer 2 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Output: (64, 8, 8)
        
        # Convolutional Layer 3 + ReLU
        x = F.relu(self.conv3(x))             # Output: (8, 2, 2)
        
        # flatten and pass into the fc
        x = x.view(x.size(0), -1)             # Flatten: (N, 8 * 2 * 2)
        x = self.fc_1(x)                      # Fully Connected Layer

        return x

        
