from vizdoom import *
import itertools as it
import random
from random import sample, randint
from time import time, sleep
import numpy as np
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
from model import DoomNet
import shutil
import math
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark=True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

learning_rate = 0.00001
discount_factor = 0.99
epochs = 200

# Training regime
training_episodes_per_epoch = 100
testing_episodes_per_epoch = 100

# Other parameters
seq_len=20
frame_repeat = 5
resolution = (120, 160)
episodes_to_watch = 10

model_dir='./save_simple'
if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)
model_loadfile = "./save/model_45.pt"
model_savefile = os.path.join(model_dir,"model.pth")
save_model = True
load_model = False

# Configuration file path
config_file_path = "../ViZDoom/scenarios/health_gathering.cfg"
#config_file_path = "../ViZDoom/scenarios/my_way_home.cfg"
#config_file_path = "../ViZDoom/scenarios/rocket_basic.cfg"
#config_file_path = "../ViZDoom/scenarios/basic.cfg"
#config_file_path = "../ViZDoom/scenarios/deadly_corridor.cfg"

# Converts and down-samples the input image
def preprocess(state):
    img = state.screen_buffer
    #img = np.moveaxis(img, [0,1,2], [2,0,1])
    img = Image.fromarray(img)
    img = Resize(75) (img)
    img = ToTensor() (img)
    #img = img.unsqueeze(0)
    #img = img.cuda()
    depth = state.depth_buffer
    depth = Image.fromarray(depth)
    depth = Resize(75) (depth)
    depth = ToTensor() (depth)
    #depth = depth.unsqueeze(0)
    img=torch.cat((img,depth),0)
    img=img.unsqueeze(0)
    img=img.cuda()
    #print(depth)
    return img

#criterion = nn.SmoothL1Loss()
criterion=nn.MSELoss()

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    #game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_depth_buffer_enabled(True)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':

    of = open(os.path.join(model_dir,'test.txt'), 'w')

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    if load_model:
        print("Loading model from: ", model_loadfile)
        model = DoomNet(len(actions))
        my_sd=torch.load(model_loadfile)
        model.load_state_dict(my_sd)
    else:
        model = DoomNet(len(actions))
    model=model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting the training!")
    time_start = time()
    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []
        health_curr=100.0
        loss_value_total=0.0; loss_policy_total=0.0; loss_entropy_total=0.0; loss_total=0.0
        steps_total=0

        print("Training...")
        model=model.train()
        for learning_step in trange(training_episodes_per_epoch, leave=False):
            t=0; loss=0.0; loss_value=0.0; loss_policy=0.0
            state=model.init_hidden()
            game.new_episode()
            while not game.is_episode_finished():
                reward_list=[]; probs_list=[]; log_probs_list=[]; value_list=[]
                loss=0.0
                for t in range(seq_len):
                    s1 = preprocess(game.get_state())#.screen_buffer)
                    (policy, value, state) = model(V(s1), state)
                    #print(value.size())
                    probs=F.softmax(policy,1)
                    log_probs=F.log_softmax(policy,1)
                    m, index = torch.max(probs.data, 1)
                    a = index[0]
                    probs_list.append(probs[0,a])
                    log_probs_list.append(log_probs[0,a])
                    #reward = game.make_action(actions[a], frame_repeat)/100.0
                    game.make_action(actions[a], frame_repeat)
                    health_prev=health_curr
                    health_curr=game.get_game_variable(GameVariable.HEALTH)
                    reward=(health_curr-health_prev)/100.0
                    #print(reward)
                    reward_list.append(reward)
                    value_list.append(value[0,0])
                    isterminal = game.is_episode_finished()
                    if isterminal:
                        break
                if isterminal:
                    R=0.0
                else:
                    s2 = preprocess(game.get_state())#.screen_buffer)
                    (_, v, _) = model(V(s2), state)
                    R = v.item()
                for i in reversed(range(len(reward_list))):
                    R=reward_list[i]+discount_factor*R
                    advantage=R-value_list[i].item()
                    loss_policy=-log_probs_list[i]*advantage
                    loss_value=criterion(value_list[i],V(torch.cuda.FloatTensor([R])))
                    #print(R)
                    #print(value_list[i].data[0])
                    loss_entropy = (-1) * (-1) * (probs_list[i] * log_probs_list[i]).sum()
                    loss += loss_policy+loss_value+0.01*loss_entropy
                    loss_policy_total += loss_policy.item()
                    loss_value_total += loss_value.item()
                    loss_entropy_total += loss_entropy.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
                optimizer.step()
                for j in range(len(state)):
                    state[j] = state[j].detach()
                    #state[j][1] = state[j][1].detach()
                steps_total += len(reward_list)
            score = game.get_total_reward()
            train_scores.append(score)

        train_scores = np.array(train_scores)
        print("Results: mean: %.2f std: %.2f," % (train_scores.mean(), train_scores.std()), "min: %.2f," % train_scores.min(), "max: %.2f," % train_scores.max())
        print('Loss_policy: %f, loss_value: %f, loss_entropy: %f' % (loss_policy_total/steps_total, loss_value_total/steps_total, loss_entropy_total/steps_total))

        print("\nTesting...")
        test_episode = []
        test_scores = []
        model=model.eval()
        for test_episode in trange(testing_episodes_per_epoch, leave=False):
            state=model.init_hidden()
            game.new_episode()
            while not game.is_episode_finished():
                s1 = preprocess(game.get_state())#.screen_buffer)
                (actual_q, _, state) = model(V(s1), state)
                m, index = torch.max(actual_q, 1)
                a = index.data[0]
                game.make_action(actions[a], frame_repeat)
            score = game.get_total_reward()
            test_scores.append(score)

        test_scores = np.array(test_scores)
        print("Results: mean: %.2f std: %.2f," % (test_scores.mean(), test_scores.std()), "min: %.2f" % test_scores.min(), "max: %.2f" % test_scores.max())

        print("Saving the network weigths to:", model_savefile)
        torch.save(model.state_dict(), model_savefile)

        print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
        of.write('%d,%f,%f\n' % (epoch + 1, (time() - time_start) / 60.0, test_scores.mean())); of.flush()

    game.close()
    print("======================================")
    print("Training finished.")


