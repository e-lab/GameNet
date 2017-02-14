# -*- coding: utf-8 -*-

import numpy as np
from itertools import count
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T


class Policy(nn.Module):
    def __init__(self, width=320, height=160, channels=3):
        super(Policy, self).__init__()
		# Matchnet here
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 7)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []
		# INPUT SIZE
		self.width = width
		self.height = height
		self.channels = channels
		
    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
		
        return F.softmax(action_scores), state_values


		 
	
	

