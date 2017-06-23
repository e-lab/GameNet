# Xinrui Wang, April 2017
# Collaborator: Eugenio
#
# Send commands(throttle and steering) to game GTA V
#

# Python imports
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,Dataset
import numpy as np
from PIL import Image
import torchvision
from torchvision.utils import save_image
import argparse
import socket, struct
import numpy as np
from array import array
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# Local imports
from opts import get_args # Get all the input arguments
from model_prednet import Prednet

print('GTAV START DRIVING')

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

args = get_args()                   # Holds all the input argument

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)

args_log = open(args.savedir + '/args.log', 'w')
args_log.write(str(args))
args_log.close()

torch.manual_seed(args.seed)        # Set random seed manually
if torch.cuda.is_available():
    if not args.cuda:
        print(CP_G + "WARNING: You have a CUDA device, so you should probably run with --cuda" + CP_C)
    else:
        torch.cuda.set_device(args.devID)
        torch.cuda.manual_seed(args.seed)
        print("\033[41mGPU({:}) is being used!!!{}".format(torch.cuda.current_device(), CP_C))

trans = torchvision.transforms.Compose([
            torchvision.transforms.Scale(args.height), torchvision.transforms.ToTensor()
                             ])
dtype = torch.cuda.FloatTensor


class Variable(Variable):
    def __init__(self, data, *args, **kwargs):
        data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def transform(num):
    filename = args.datadir+str(num)+ '.png'
    im = Image.open(filename)
    return trans(im)#.unsqueeze(0)

class MyDataset(Dataset):
    def __init__(self, len = args.seqLen, max = args.fileNum):
        self.length = len
        self.max = max
        self.batch = args.batchSize
        self.fname = args.datadir+'dataset.txt'
        self.target = np.loadtxt(self.fname)
        self.target_mean = self.target.mean(axis=0)
        self.target_std = self.target.std(axis=0)
        self.targetdata = (self.target[0:self.max,1:8]-self.target_mean[1:8])/self.target_std[1:8]
        self.rand = np.arange(int(self.max/self.length))*self.length+1
        self.counter = 0

    def shuffle(self,):     
        np.random.shuffle(self.rand)
        self.counter = 0

    def get_sensorreading(self, output):
        pass
        # return output*self.target_std+self.target_mean

    def get(self, idx):
        traindata = torch.zeros(self.length, args.channels, args.height, args.width)
        index = 0
        for i in range(int(idx), int(self.length+idx)):
            img = transform(i)
            img = img[0:3]
            traindata[index-1] = img
            index +=1
        target = torch.from_numpy(self.targetdata[int(idx):int(idx+self.length)])
        # save_image(traindata,'train.png')
        return traindata, target

    def getbatch(self):     
        inputd = torch.zeros(self.length, self.batch, args.channels, args.height, args.width)
        target = torch.zeros(self.length, self.batch, args.sensors)
        idx = 0
        for i in self.rand[ self.counter*self.batch : (self.counter+1)*self.batch ]:
            inputd[:,idx,:,:,:], target[:,idx,:] = self.get(i)
            idx += 1
        self.counter += 1
        # print(self.rand[self.counter*self.batch])
        return Variable(inputd.cuda()), Variable(target.cuda())

class Server:
	def __init__(self, port=8000, image_size=(200,66)):
		print('Started server')
		self.image_size = image_size
		self.buffer_size = image_size[0]*image_size[1]*3;
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.bind(('0.0.0.0', port))
		self.s.listen(1)

		self.conn, self.addr = self.s.accept()
		print('GTAV connected')

	def recvImage(self):
		data = b""
		while len(data) < self.buffer_size:
			packet = self.conn.recv(self.buffer_size - len(data))
			if not packet: return None
			data += packet

		return np.resize(np.fromstring(data, dtype='uint8'), (self.image_size[1], self.image_size[0], 3)).astype('float32')

	def sendCommands(self, throttle, steering):		
		data = array('f', [throttle, steering])
		self.conn.sendall(data.tobytes())
		print('Sent commands', data)

	def recvReward(self):
		data = b""
		while len(data) < 4:
			packet = self.conn.recv(self.buffer_size - len(data))
			if not packet: return None
			data += packet

		print('Received reward')
		return struct.unpack('f', data)[0]

	def close(self):
		self.s.close()

if __name__ == '__main__':
	net = Prednet(2,1,3, batch=args.batchSize, seq=args.seqLen)
    net.optimizer = optim.Adam([
                weights for dic in net.params for weights in dic.values()
            ])#, lr=args.lr)#, momentum=args.eta)
	net.dset = MyDataset(net.T)
	net.max_epoch = args.epochs
	net = net.cuda()
    	
	x = np.zeros((args.seqLen, args.height, args.width, 3), dtype='float32')
	
	server = Server(port=args.port, image_size=(args.width, args.height))
	while 1:
		img = server.recvImage() #receive image
		if (img == None): break
		x = np.roll(x,-1, axis=0)
		x[-1] = img 
		# input a sequence of images
		prediction = net.run(x)
		server.sendCommands(prediction[3],prediction[4])
		#throttle & direction
		reward = server.recvReward()
		if (reward == None): break
		print(reward)

	server.close()


   
