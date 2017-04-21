# Eugenio Culurciello, April 2017
# Code original: XinRui Wang
#
# train to predict sensor reading from Grand Theft Auto V 
# new model from EUGE
#

# Python imports
import os, os.path
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,Dataset
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
import torchvision
from torchvision.utils import save_image

# Local imports
from opts import get_args # Get all the input arguments


print('GTAV TRAINING TO PREDICT SENSOR DATA -- ECNet version')

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


def loadntransform(num):
    # print(num)
    filename = args.datadir+str(num)+ '.png'
    im = Image.open(filename)
    return trans(im)


class MyDataset(Dataset):
    def __init__(self, seqLen = args.seqLen, files = args.fileNum):
        self.seqLen = seqLen
        self.length = files
        self.fname = args.datadir+'/dataset.txt'
        self.target = np.loadtxt(self.fname)
        self.target_mean = self.target.mean(axis=0)
        self.target_std = self.target.std(axis=0)
        self.targetdata = (self.target[0:self.length-1,1:8]-self.target_mean[1:8])/self.target_std[1:8]
        self.counter = 0

    def __len__(self):
        l = int(self.length/self.seqLen/args.batchSize)*args.batchSize # we only take full batches to avoid crashes
        print('Data size:', l, 'number of batches to run:', l/args.batchSize)
        return l

    def __getitem__(self, index):     
        inputd = torch.zeros(self.seqLen, args.channels, args.height, args.width)
        tmp = 0
        for i in range(index, self.seqLen+index):
            # print('range', i)
            img = loadntransform(i+1)
            inputd[tmp] = img[0:3]
            tmp +=1
        target = torch.from_numpy(self.targetdata[index:self.seqLen+index]).float()
        return inputd, target



class ECnet(nn.Module):
    def __init__(self):
        super(ECnet, self).__init__()
        
        self.seqLen = args.seqLen
        self.batch_size = args.batchSize
        self.features = [3,32,64,128,256,512]
        self.c1 = nn.Conv2d( self.features[0], self.features[1], 7, padding=0 )
        self.c2 = nn.Conv2d( self.features[1], self.features[2], 5, padding=0 )
        self.c3 = nn.Conv2d( self.features[2], self.features[3], 3, padding=0 )
        # self.c4 = nn.Conv2d( self.features[3], self.features[4], 3, padding=0 )
        # self.c5 = nn.Conv2d( self.features[4], self.features[5], 3, padding=0 )
        self.avgpool = nn.AvgPool2d(1,3) # 5 if all layers are used!
        self.rnn1 = nn.RNN(self.features[3], self.features[3], args.rnnLayers)
        self.classifier1 = nn.Linear(self.features[3], args.sensors)


    def init_hidden(self):
        return Variable( torch.zeros(args.rnnLayers, self.batch_size, self.features[3]).cuda() )

    def forward(self, x, h):
        # print('input', x.size())
        pool = 4
        otot = Variable( torch.zeros(self.seqLen, self.batch_size, self.features[3]).cuda() )
        for t in range(0, self.seqLen):
            outp = F.relu(F.max_pool2d(self.c1(x[:,t]), pool, pool))
            # print('outp', outp.size())
            outp = F.relu(F.max_pool2d(self.c2(outp), pool, pool))
            # print('outp', outp.size())
            outp = F.relu(F.max_pool2d(self.c3(outp), pool, pool))
            # print('outp', outp.size())
            # outp = F.relu(F.max_pool2d(self.c4(outp), pool, pool))
            # print('outp', outp.size())
            # outp = F.relu(F.max_pool2d(self.c5(outp), pool, pool))
            # print('outp', outp.size())
            outp = self.avgpool(outp)
            # print('outp', outp.size())
            # print('otot[t]', otot[t].size())
            otot[t] = outp

        # print(otot.size(), h.size())
        y, h = self.rnn1(otot, h)
        # print('final out', y.size())
        return self.classifier1(y[self.seqLen-1]), h


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(datapoints, net):
    h = net.init_hidden()
    print('\n---------- Train a ECnet neural network ----------')
    
    # Error logger
    logger_bw = open(args.savedir + '/error_bw.log', 'w')
    logger_bw.write('{:10}'.format('Train Error'))
    logger_bw.write('\n{:-<10}'.format(''))

    print('Running for', net.max_epoch, 'Epoch')
    
    for epoch in range(0, net.max_epoch):
        net.train()
        # pbar = trange(datapoints, desc='Epoch {:03}'.format(epoch))

        for batch, (input_sequence, target_sensors) in enumerate(train_loader):
            # print('input sequence', input_sequence.size())
            # print('target sensors', target_sensors.size())
            if args.cuda:  # Convert into CUDA tensors
                input_sequence = input_sequence.cuda()
                target_sensors = target_sensors.cuda()
            input_sequence = Variable(input_sequence) # convert to Variable
            target_sensors = Variable(target_sensors)
            net.optimizer.zero_grad() # zero the gradient buffers
            loss = 0
            h = repackage_hidden(h)
            predictions, h = net.forward(input_sequence, h)
            # print('predictions size', predictions.size())
            loss = net.criterion(predictions, target_sensors[:,net.seqLen-1])
            loss.backward()
            net.optimizer.step()
            print(' >>> Epoch {:d}, Batch {:2d}, sensor_loss: {:.3f}'.format( epoch+1, batch + 1, loss.data[0]))
            if batch % 10 == 0:
                logger_bw.write('\n{:.6f}'.format(loss.data[0]))
                # pbar.set_postfix(str=' >>> Batch {:2d}, sensor_loss: {:.3f}'.format((batch + 1), loss.data[0]))
                # pbar.referesh()
                # pbar.update(10)
        
        net.eval()
        torch.save(net.state_dict(), args.savedir +'ECNet_weights')
        torch.save(net, args.savedir +'ECNet_net')   

    pbar.close()
    print('Finished training!')

if __name__ == '__main__':
    net = ECnet()
    net = net.cuda()
    net.criterion = nn.MSELoss()
    net.optimizer = optim.Adam( params=net.parameters() ) #, lr=args.lr, momentum=args.eta)
    args.fileNum = len(os.listdir(args.datadir))-1

    # create dataset and loaders:  
    train_loader = DataLoader(dataset=MyDataset(seqLen=args.seqLen, files=args.fileNum), 
                num_workers=args.threads, 
                batch_size=args.batchSize, shuffle=True)

    net.max_epoch = args.epochs
    train( datapoints=args.fileNum, net=net)
    # model = torch.nn.DataParallel(net,device_ids=[0, 1, 2])
    # model.train()
