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
        self.relu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        #self.fc1 = nn.Linear(32 * 3 * 3, 1024)
        #self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTMCell(32 * 3 * 3, 256)
        self.fc_val = nn.Linear(256, 1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, state):
        hx1i=state[0]
        cx1i=state[1]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
   
        x=x.view(1, 32 * 3 * 3)
        #x=self.dropout(self.relu(self.fc1(x)))
        (hx1o,cx1o)=self.lstm1(x,(hx1i,cx1i))
        #hx1o=self.dropout(hx1o); cx1o=self.dropout(cx1o)
        v=self.fc_val(hx1o)
        y=self.fc(hx1o)

        state=[hx1o,cx1o]

        return (y, v, state)

    def init_hidden(self):
        state=[]
        state.append(torch.zeros(1, 256).cuda())
        state.append(torch.zeros(1, 256).cuda())
        return state

class ICM(nn.Module):
    def __init__(self, num_classes):
        super(ICM, self).__init__()
        self.num_classes = num_classes
        self.relu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        self.inverse_fc1 = nn.Linear(288 * 2, 256)
        self.inverse_fc2 = nn.Linear(256, num_classes)
        self.forward_fc1 = nn.Linear(288 + num_classes, 256)
        self.forward_fc2 = nn.Linear(256, 288)

    def forward(self, x1, x2, a):
        
        a_in = torch.zeros(1, self.num_classes).cuda()
        a_in[0, a] = 1.0

        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        x1 = self.relu(self.conv4(x1))
        emb1 = x1.view(1, 32 * 3 * 3)

        x2 = self.relu(self.conv1(x2))
        x2 = self.relu(self.conv2(x2))
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        emb2 = x2.view(1, 32 * 3 * 3)
        
        x = torch.cat((emb1, emb2), 1)
        x = self.relu(self.inverse_fc1(x))
        a_out = self.inverse_fc2(x)

        x = torch.cat((emb1.detach(), a_in), 1)
        x = self.relu(self.forward_fc1(x))
        emb2_out = self.forward_fc2(x)

        return (a_out, emb2_out, emb2.detach())

