# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
from critic import Critic
from actor import Actor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T
from server import Server
from OU import OU
from memory import ReplayBuffer

TRAIN_FLAG = 1
MAXIMUM_MEMORY = 100000
CHANNEL = 3
lr_actor = 0.0001
lr_critic = 0.001
action_dim = 2 # steer and throttle
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()
STEPS = 100000
resize = T.Compose([T.ToTensor()])# need fixing
OU = OU() #Ornstein-Uhlenbeck Process

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
		
def get_screen():
	img = server.recvImage()
	
	return resize(img).unsqueeze(0)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Reinforcement traning')
	parser.add_argument('port', type=int, help='Port to listen to')
	parser.add_argument('width', type=int, help='Width of the image to receive')
	parser.add_argument('height', type=int, help='Height of the image to receive')
	args = parser.parse_args()
	
	# Create model and optimizer
	actor = Actor(args.width,args.height,CHANNEL)
	actor_target = Actor(args.width,args.height,CHANNEL)
	critic = Critic(args.width,args.height,CHANNEL)
	critic_target = Critic(args.width,args.height,CHANNEL)
	optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
	optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
	
	if USE_CUDA:
		actor.cuda()
		critic.cuda()
		critic_target.cuda()
		actor_target.cuda()
	
	# create memory
	replay = ReplayBuffer(MAXIMUM_MEMORY)
	
	#initiate server
	server = Server(port=args.port, image_size=(args.width, args.height))
	
	while 1:
		for t in range(STEPS): # Don't infinite loop while learning
            epsilon = EPS_START - (EPS_START - EPS_END)/EPS_DECAY*t #update episilon
            a_t = torch.zeros([1,action_dim]) #[[throttle,steer]]
            noise_t = torch.zeros([1,action_dim])
			s_t = get_screen()
			
			a_t_original = actor(s_t.reshape(1, s_t.shape[0])) #make action predictions
			noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.5 , 1.00, 0.10)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.0 , 0.60, 0.30)
			a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
			
			server.sendCommands(a_t[0,0], a_t[0,1])
			s_t1 = get_screen()
			if (img == None): break
			reward = server.recvReward()
			if (reward == None): break
			print(reward)
			Replay.add(s_t, a_t[0], reward, s_t1) # Remember
			
			# Train the networks
			# get batch
			batch = buff.getBatch(BATCH_SIZE)
            states = Variable([e[0] for e in batch])
            actions = Variable([e[1] for e in batch])
            rewards = Variable([e[2] for e in batch])
            new_states = Variable([e[3] for e in batch])
			
			state_action_values = critic([states, actions])
			target_q_values = critic([new_states, actor(new_states)])  
			expected_state_action_values = Variable(torch.zeros(BATCH_SIZE))
			# clear the volatile flag
			target_q_values.volatile = False
			
			for k in range(len(batch)):
				# SARSA updating rule
				expected_state_action_values_state_values[k] = rewards[k] + GAMMA*target_q_values[k]
			
			if (TRAIN_FLAG):
				loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
				# Optimize the model
				optimizer_critic.zero_grad()
				loss.backward()
				for param in critic.parameters():
					param.grad.data.clamp_(-1, 1)
				optimizer_critic.step()
				
				a_for_grad = actor(states)
				grads = ##critic.gradients(states, a_for_grad)
				# actor.train(states, grads) training using gradients
				# actor_target.train()
                # critic_target.train()
				
			
	server.close()


