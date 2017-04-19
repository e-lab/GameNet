# Eugenio Culurciello, April 2017
# Code original: XinRui Wang
#
# train to predict sensor reading from Grand Theft Auto V 
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
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# Local imports
from opts import get_args # Get all the input arguments
from model_prednet import Prednet

print('GTAV TRAINING TO PREDICT SENSOR DATA')
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
            torchvision.transforms.Scale(128), torchvision.transforms.ToTensor()
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
        self.rand = np.arange(int(self.max/self.length))
        self.rand = self.rand*self.length+1
        self.counter = 0

    def shuffle(self,):     
        np.random.shuffle(self.rand)
        self.counter = 0

    def get_sensorreading(self, output):
        pass
        # return output*self.target_std+self.target_mean

    def get(self, idx):
        traindata = torch.zeros(self.length,3,128,256)
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
        inputd = torch.zeros(self.length,self.batch,3,128,256)
        target = torch.zeros(self.length,self.batch,7)
        idx = 0
        for i in self.rand[ self.counter*self.batch : (self.counter+1)*self.batch ]:
            inputd[:,idx,:,:,:], target[:,idx,:] = self.get(i)
            idx += 1
        self.counter += 1
        # print(self.rand[self.counter*self.batch])
        return Variable(inputd.cuda()), Variable(target.cuda())




def save_weight(net):
    torch.save(net.params, 'prednet_weights')
    print("Saved weights!")


def train(net):
    target_sequence = Variable(torch.zeros(net.T, net.batch_size, net.output_channels, net.height_ratio * 2 ** 7, net.width_ratio * 2 ** 7).cuda())
    input_sequence,target_parameter = net.dset.getbatch()
    # target_sequence = Variable(torch.zeros(input_sequence.size()))
    print('\n---------- Train a '+str(net.number_of_layers)+' layer network ----------')
    print('Input has size', input_sequence.size())
    print('Create a MSE criterion')

    print('Run for', net.max_epoch, 'Epoch')
    
    for epoch in range(0, net.max_epoch):
        net.dset.shuffle()
        #totloss = 0
    
        save_weight(net)

        for batch in range(0, int(args.fileNum/net.T/net.batch_size)-1):
            input_sequence, target_parameter = net.dset.getbatch()
            net.optimizer.zero_grad() # zero the gradient buffers
            loss = 0
            loss1 = 0
            loss2 = 0
            error = [Variable(torch.zeros(net.batch_size,2*net.Ahat_size_list[l],net.height_ratio*2 **(net.exp-l),net.width_ratio*2 **(net.exp-l)).cuda())
                for l in range(0,net.number_of_layers)]
            state = [(
                Variable(torch.zeros(net.batch_size, net.R_size_list[l], net.height_ratio*2 **(net.exp-l),net.width_ratio*2 **(net.exp-l)).cuda()),
                Variable(torch.zeros(net.batch_size, net.R_size_list[l],net.height_ratio*2 **(net.exp-l), net.width_ratio*2 **(net.exp-l)).cuda())
                ) for l in range(0,net.number_of_layers)]
            for t in range(0, net.T):
                # repackage
                for l in range(0,net.number_of_layers):
                    state[l] = (Variable(state[l][0].data),Variable(state[l][1].data))
                    error[l] = Variable(error[l].data)
                error, state, predict_parameter = net.forward(input_sequence[t], error, state,)
                # if t != 0:
                #     ax = fig.add_subplot( 111 )
                #     im = ax.imshow(np.zeros((128, 256*2, 3)))
                #     img=mpimg.imread(str(t)+'.png')
                #     im.set_data(img)
                #     param = "Steering:%f\nThrottle:%f" % (target_parameter[t][3],target_parameter[t][4])
                #     txt1 = ax.text(20,30,speed,style='italic',
                #         bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
                #     param_real = "Steering:%f\nThrottle:%f" % (predict_parameter[t][3],predict_parameter[t][4])
                #     txt2 = ax.text(20,158,speed,style='italic',
                #         bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
                #     plt.savefig(str(t)+'.png')
                if t != 0:
                    loss1 += net.loss_fn(error[0], target_sequence[t])
                    loss2 += net.loss_fn(predict_parameter,target_parameter[t])
            
            print(' >>> Batch {:2d} image_loss: {:.3f} sensor_loss: {:.3f}'.format((batch + 1), loss1.data[0],loss2.data[0]))
            #totloss = totloss + loss
            loss = loss1+loss2
            loss.backward()
            net.optimizer.step()
            # if batch % 100 == 0:
            #     net.save_weight()
            # print(' > Epoch {:2d} loss: {:.3f}'.format((batch + 1), loss.data[0]))
            
            

if __name__ == '__main__':
    net = Prednet(2,1,3, batch=args.batchSize, seq=args.seqLen)
    net.optimizer = optim.SGD([
                weights for dic in net.params for weights in dic.values()
            ], lr=args.lr, momentum=args.eta)
    net.dset = MyDataset(net.T)
    net.max_epoch = args.epochs
    net = net.cuda()
    train(net)
    # model = torch.nn.DataParallel(net,device_ids=[0, 1, 2])
    # model.train()
