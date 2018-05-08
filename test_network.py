#!/usr/bin/env python
# -*- coding: utf-8 -*-

# E. Culurciello
# August 2017

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import random
from random import sample, randint
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable as V
from tqdm import trange
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import cv2
import shutil
import math
from model import DoomNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.benchmark=True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Q-learning settings
learning_rate = 0.00001
discount_factor = 0.99
epochs = 200
learning_steps_per_epoch = 2000
replay_memory_size = 10000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10

# NN learning settings
batch_size = 64

# Training regime
episodes_per_epoch = 1000

# Other parameters
frame_repeat = 5
resolution = (120, 160)
episodes_to_watch = 10

model_loadfile = "./save/model.pth"

# Configuration file path
#config_file_path = "../ViZDoom/scenarios/my_way_home.cfg"
config_file_path = "../ViZDoom/scenarios/health_gathering_supreme.cfg"
#config_file_path = "../ViZDoom/scenarios/basic.cfg"
#config_file_path = "../ViZDoom/scenarios/defend_the_center.cfg"

# Converts and down-samples the input image
def preprocess(img):
    img = np.moveaxis(img, [0,1,2], [2,0,1])
    img = Image.fromarray(img)
    img = Resize(75) (img)
    img = ToTensor() (img)
    #img = img.view(1,3,100,120)
    img = img.unsqueeze(0)
    img = img.cuda()
    return img

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

if __name__ == '__main__':

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    print("Loading model from: ", model_loadfile)
    model = DoomNet(len(actions))
    my_sd=torch.load(model_loadfile)
    model.load_state_dict(my_sd)
    model=model.cuda()
    model=model.eval()

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))

    count_fr=0
    all_scores=np.zeros((episodes_to_watch),dtype=np.float32)
    #new_video = np.zeros((2000, 480, 640, 3), dtype=np.uint8)
    for j in range(episodes_to_watch):
        game.new_episode()
        count_fr=0
        state=model.init_hidden()
        while not game.is_episode_finished():
            s1 = preprocess(game.get_state().screen_buffer)

            #out.write(frame)
            #new_video[count_fr]=frame
            #count_fr+=1

            (actual_q, _, state) = model(s1, state)
            m, index = torch.max(actual_q, 1)
            a = index.item()

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[a])
            for _ in range(frame_repeat):
                #if not game.is_episode_finished():
                #    s1, frame = preprocess(game.get_state().screen_buffer)
                    #out.write(frame)
                game.advance_action()
                sleep(0.02)

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        all_scores[j]=score
        print("Total score: ", score)

    final_score=all_scores.mean()
    print('Final scores is ', final_score)

