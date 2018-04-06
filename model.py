import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable as V
from math import ceil
import time
import copy
from torchvision.utils import save_image
import os
import shutil
import argparse

class DoomNet(nn.Module):
    def __init__(self, num_classes):
        super(DoomNet,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3,32, kernel_size=8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.fc1 = nn.Linear(64*10*13,1024)
        self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTMCell(1024,1024)
        self.fc_val = nn.Linear(1024,1)
        self.fc = nn.Linear(1024,num_classes)

    def forward(self, x, state):
        hx1i=state[0]
        cx1i=state[1]

        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
   
        x=x.view(1,64*10*13)
        x=self.dropout(self.relu(self.fc1(x)))
        (hx1o,cx1o)=self.lstm1(x,(hx1i,cx1i))
        hx1o=self.dropout(hx1o); cx1o=self.dropout(cx1o)
        v=self.fc_val(hx1o)
        y=self.fc(hx1o)

        state=[hx1o,cx1o]

        return (y, v, state)

    def init_hidden(self):
        state=[]
        state.append(torch.zeros(1,1024).cuda())
        state.append(torch.zeros(1,1024).cuda())
        return state


