"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Source(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)  # SAME padding
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)  # SAME padding
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, stride=2, padding=2)  # SAME padding
        self.fc1 = nn.Linear(in_features=8 * 2 * 2, out_features=8)  # Fully connected layer

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            ## TODO: initialize the parameters for the convolutional layers
            nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1 / (5 * 5 * conv.in_channels)))
            nn.init.constant_(conv.bias, 0.0)
        
        ## TODO: initialize the parameters for [self.fc1]
        nn.init.normal_(self.fc1.weight, mean=0.0, std=sqrt(1 / self.fc1.in_features))
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape

        ## TODO: forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


