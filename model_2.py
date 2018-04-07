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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(256*8*10,1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_val = nn.Linear(1024,1)
        self.fc = nn.Linear(1024,num_classes)

    def forward(self, x, state):
        hx1=state[0]
        hx2=state[1]
        hx3=state[2]

        x=self.relu(self.conv1(x))
        state[0]=x
        x=torch.cat((x,hx1),1)
        x=self.relu(self.conv2(x))
        state[1]=x
        x=torch.cat((x,hx2),1)
        x=self.relu(self.conv3(x))
        state[2]=x
        x=torch.cat((x,hx3),1)
        x=self.relu(self.conv4(x))   

        x=x.view(1,256*8*10)
        x=self.dropout(self.relu(self.fc1(x)))
        #(hx1o,cx1o)=self.lstm1(x,(hx1i,cx1i))
        #hx1o=self.dropout(hx1o); cx1o=self.dropout(cx1o)
        v=self.fc_val(x)
        y=self.fc(x)

        #state=[hx1o,cx1o]

        return (y, v, state)

    def init_hidden(self):
        state=[]
        state.append(torch.zeros(1,32,60,80).cuda())
        state.append(torch.zeros(1,64,30,40).cuda())
        state.append(torch.zeros(1,128,15,20).cuda())
        state.append(torch.zeros(1,256,8,10).cuda())
        return state


