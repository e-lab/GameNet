from vizdoom import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.utils import save_image
from PIL import Image
import cv2
import shutil
import math
from argparse import ArgumentParser
import os
import itertools as it
import random
from random import sample, randint
from time import time, sleep
from tqdm import trange
from model import DoomNet
from model import ICM

parser = ArgumentParser()
_ = parser.add_argument
_('--save_dir', type = str, default = './save', help = 'Save directory')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.benchmark=True
random.seed(0)
torch.manual_seed(0)

learning_rate = 0.0001
discount_factor = 0.99
epochs = 1000
training_episodes_per_epoch = 80
testing_episodes_per_epoch = 20
seq_len = 20
frame_repeat = 4
resolution = [42, 42]
reward_scaling = 1.0
reward_intrinsic_scaling = 100.0
value_loss_scaling = 0.5
entropy_loss_scaling = 0.01
max_grad = 40.0
#config_file_path = "../ViZDoom/scenarios/health_gathering_supreme.cfg"
#config_file_path = "../ViZDoom/scenarios/my_way_home.cfg"
config_file_path = "./scenarios/my_way_home_sparse.cfg"
#config_file_path = "../ViZDoom/scenarios/rocket_basic.cfg"
#config_file_path = "../ViZDoom/scenarios/basic.cfg"
#config_file_path = "../ViZDoom/scenarios/deadly_corridor.cfg"
load_model = False
model_dir = args.save_dir
model_loadfile = "./save/model.pth"
model_savefile = os.path.join(model_dir, "model.pth")

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    #game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_depth_buffer_enabled(False)
    game.init()
    print("Doom initialized.")
    return game

def preprocess(state):
    img = state.screen_buffer
    img = np.moveaxis(img, [0,1,2], [2,0,1])
    img = Image.fromarray(img)
    img = Resize(resolution) (img)
    img = ToTensor() (img)
    #depth = state.depth_buffer
    #depth = Image.fromarray(depth)
    #depth = Resize(120) (depth)
    #depth = ToTensor() (depth)
    #img=torch.cat((img,depth),0)
    img=img.unsqueeze(0)
    img=img.cuda()
    return img

if __name__ == '__main__':

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    of = open(os.path.join(model_dir, 'test.txt'), 'w')

    game = initialize_vizdoom(config_file_path)
    n = game.get_available_buttons_size()
    #actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]

    if load_model:
        print("Loading model from: ", model_loadfile)
        model = DoomNet(len(actions))
        my_sd = torch.load(model_loadfile)
        model.load_state_dict(my_sd)
    else:
        model = DoomNet(len(actions))
    model_icm = ICM(len(actions))
    model=model.cuda()
    model_icm = model_icm.cuda()
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer_icm = torch.optim.Adam(model_icm.parameters(), lr = learning_rate)

    print("Starting the training!")
    time_start = time()
    training_steps = 0
    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []        
        loss_value_total = 0.0
        loss_policy_total = 0.0
        loss_entropy_total = 0.0    
        loss_total = 0.0
        loss_inverse_total = 0.0
        loss_forward_total = 0.0
        loss_icm_total = 0.0
        reward_intrinsic_total = 0.0
        steps_total=0

        print("Training...")
        model.train()
        for learning_step in trange(training_episodes_per_epoch, leave=False):
            t = 0
            loss = 0.0
            loss_icm = 0.0
            hidden = model.init_hidden()
            game.new_episode()

            inp = preprocess(game.get_state())
            inp1 = inp
            (policy, value, hidden) = model(inp, hidden)
            probs = F.softmax(policy,1)
            a = probs.multinomial(num_samples=1).detach().item()
            a_icm = a
            game.make_action(actions[a], frame_repeat)
            for i in range(len(hidden)):
                hidden[i] = hidden[i].detach()
            isterminal = game.is_episode_finished()
            if isterminal:
                continue
            
            while not game.is_episode_finished():
                probs_list=[]
                log_probs_list=[]
                value_list=[]
                reward_list=[]
                loss=0.0
                loss_icm = 0.0
            
                for t in range(seq_len):
                    inp = preprocess(game.get_state())
                    (policy, value, hidden) = model(inp, hidden)
                    probs = F.softmax(policy,1)
                    log_probs = F.log_softmax(policy,1)
                    #m, index = torch.max(policy, 1)
                    #a = index.item()
                    a = probs.multinomial(num_samples=1).detach().item()
                    probs_list.append(probs[0, a])
                    log_probs_list.append(log_probs[0, a])
                    value_list.append(value)
                    reward = game.make_action(actions[a], frame_repeat) / reward_scaling
            
                    inp2 = inp
                    a_out, emb_out, emb = model_icm(inp1, inp2, a_icm)       
                    a_label = torch.LongTensor([a_icm]).cuda()
                    reward_intrinsic = (emb_out - emb).pow(2).mean().item() / reward_intrinsic_scaling
                    reward += reward_intrinsic
                    reward = min(reward, 1.0)
                    #print(reward_intrinsic)
                    loss_inverse = nll(a_out, a_label)
                    loss_forward = criterion(emb_out, emb)
                    loss_icm = 0.8 * loss_inverse + 0.2 * loss_forward 
                    loss_inverse_total += loss_inverse.item()
                    loss_forward_total += loss_forward.item()
                    reward_intrinsic_total += reward_intrinsic
                    inp1 = inp2        
                    a_icm = a             
                    optimizer_icm.zero_grad()
                    loss_icm.backward()
                    torch.nn.utils.clip_grad_norm_(model_icm.parameters(), max_grad)
                    optimizer_icm.step()
                    
                    #print(reward)
                    reward_list.append(reward)           
                    isterminal = game.is_episode_finished()
                    if isterminal:
                        break
                if isterminal:
                    R=0.0
                else:
                    inp = preprocess(game.get_state())
                    (_, value, _) = model(inp, hidden)
                    R = value.item()
                value_list.append(torch.FloatTensor([[R]]).cuda())
                gae=0.0
                for i in reversed(range(len(reward_list))):
                    R = reward_list[i] + discount_factor * R
                    #print(R)
                    #advantage=R-value_list[i].item()
                    delta_t = reward_list[i] + discount_factor * value_list[i+1].item() - value_list[i].item()
                    gae = discount_factor * gae + delta_t
                    loss_policy = -log_probs_list[i] * gae #*advantage
                    loss_value = criterion(value_list[i], torch.FloatTensor([[R]]).cuda())
                    loss_entropy = (-1) * (-1) * (probs_list[i] * log_probs_list[i]).sum()
                    loss += loss_policy + value_loss_scaling * loss_value + entropy_loss_scaling * loss_entropy
                    loss_policy_total += loss_policy.item()
                    loss_value_total += loss_value.item()
                    loss_entropy_total += loss_entropy.item()
                optimizer.zero_grad()
                loss *= 0.1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
                for i in range(len(hidden)):
                    hidden[i] = hidden[i].detach()
                steps_total += len(reward_list)
            score = game.get_total_reward()
            train_scores.append(score)

        train_scores = np.array(train_scores)
        training_steps += steps_total
        print("Results: mean: {:.2f} std: {:.2f}, min: {:.2f}, max: {:.2f}".format(train_scores.mean(), train_scores.std(), train_scores.min(), train_scores.max()))
        print('Loss_policy: {:f}, loss_value: {:f}, loss_entropy: {:f}'.format(loss_policy_total/steps_total, loss_value_total/steps_total, loss_entropy_total/steps_total))
        print('Reward intrinsic: {:f}, Loss_inverse: {:f}, loss_forward: {:f}'.format(reward_intrinsic_total/steps_total, loss_inverse_total/steps_total, loss_forward_total/steps_total))


        with torch.no_grad():
            print("\nTesting...")
            test_scores = []
            model.eval()
            for test_episode in trange(testing_episodes_per_epoch, leave=False):
                hidden = model.init_hidden()
                game.new_episode()
                while not game.is_episode_finished():
                    inp = preprocess(game.get_state())
                    (policy, _, hidden) = model(inp, hidden)
                    m, index = torch.max(policy, 1)
                    a = index.item()
                    game.make_action(actions[a], 4)
                score = game.get_total_reward()
                test_scores.append(score)

            test_scores = np.array(test_scores)
            print("Results: mean: {:.2f} std: {:.2f}, min: {:.2f}, max: {:.2f}".format(test_scores.mean(), test_scores.std(), test_scores.min(), test_scores.max()))

        print("Saving the network weigths to:", model_savefile)
        torch.save(model.state_dict(), model_savefile)

        print("Total training steps: {:d}, Total elapsed time: {:.2f} minutes".format(training_steps, (time() - time_start) / 60.0))
        of.write('{:d},{:d},{:f},{:f}\n'.format(epoch + 1, training_steps, (time() - time_start) / 60.0, test_scores.mean()))
        of.flush()

    game.close()
    print("======================================")
    print("Training finished.")


