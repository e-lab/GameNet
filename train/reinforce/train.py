# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
from model import Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T
from server import Server

SavedAction = namedtuple('SavedAction', ['action', 'value'])
MAXIMUM_MEMORY = 1000

def image_process(image):
	return image

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    action = probs.multinomial()
	# Save action and Qvalue to the "memory" of network
	model.saved_actions.append(SavedAction(action, state_value))
	# Forget actions a long time ago
	while len(model.saved_actions) > MAXIMUM_MEMORY:
		model.saved_actions.pop(0)
    return action.data

def train_network(): # need fixing
    R = 0
    saved_actions = model.saved_actions
    value_loss = 0
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for (action, value), r in zip(saved_actions, rewards):
        action.reinforce(r - value.data.squeeze())
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))
    optimizer.zero_grad()
    final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
    gradients = [torch.ones(1)] + [None] * len(saved_actions)
    autograd.backward(final_nodes, gradients)
    optimizer.step()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Reinforcement traning')
	parser.add_argument('port', type=int, help='Port to listen to')
	parser.add_argument('width', type=int, help='Width of the image to receive')
	parser.add_argument('height', type=int, help='Height of the image to receive')
	args = parser.parse_args()
	
	# Create model and optimizer
	model = Policy()
	optimizer = optim.Adam(model.parameters(), lr=3e-2)
	
	#initiate server
	server = Server(port=args.port, image_size=(args.width, args.height))
	
	while 1:
		
		for t in range(10000): # Don't infinite loop while learning
			img = server.recvImage()
			if (img == None): break
			commands = select_action(image_process(img))
			server.sendCommands(commands[0,0], commands[0,1])
			reward = server.recvReward()
			if (reward == None): break
			model.rewards.append(reward)
			while len(model.rewards) > MAXIMUM_MEMORY:
				model.rewards.pop(0)
			print(reward)
		
			train_network()
			
	server.close()


