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
import torchvision.transforms as T
from torchvision.utils import save_image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#torch.manual_seed(0)
torch.cuda.set_device(0)

USE_CUDA = torch.cuda.is_available()
if USE_CUDA ==0:
    print "WARNING:CUDA IS NOT AVAILABLE"
trans=T.Compose([
            T.Scale(128),T.ToTensor()
                             ])
dtype = torch.cuda.FloatTensor

class Variable(Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def transform(num):
    filename = '/media/HDD1/Datasets/GTA/2/'+str(num)+ '.png'
    im = Image.open(filename)
    return trans(im)#.unsqueeze(0)

class MyDataset(Dataset):
    def __init__(self,len = 10,batch = 10,max=30116,):
        self.length = len
        self.max = max
        self.batch = batch
        self.fname = '/media/HDD1/Datasets/GTA/2/dataset.txt'
        self.target = np.loadtxt(self.fname)
        self.target_mean = np.array([12.3,-0.0218,0.224,0.007117,0.268,-0.5638,-0.042])
        self.target_std = np.array([9.54,3.18,0.404,0.0817,0.3564,7.8035,0.8895])
        self.targetdata = torch.from_numpy((self.target[0:max,1:8]-self.target_mean)/self.target_std)
        #self.targetdata = self.targetdata[0:max,1:8]
        self.rand = np.arange(self.max/self.length)
        self.rand = self.rand*self.length+1
        self.counter = 0

    def shuffle(self,):     
        np.random.shuffle(self.rand)
        self.counter = 0

    def get_sensorreading(self,output):
        pass
        # return output*self.target_std+self.target_mean

    def get(self, idx):
        traindata = torch.zeros(self.length,3,128,256)
        index = 0
        for i in range(idx,self.length+idx):
            img = transform(i)
            img = img[0:3]
            traindata[index-1] = img
            index +=1
        target = self.targetdata[idx:idx+self.length]
        # save_image(traindata,'train.png')
        return traindata,target

    def getbatch(self):
        
        inputd = torch.zeros(self.length,self.batch,3,128,256)
        target = torch.zeros(self.length,self.batch,7)
        idx = 0
        for i in self.rand[self.counter*self.batch:(self.counter+1)*(self.batch)]:
            inputd[:,idx,:,:,:],target[:,idx,:] = self.get(i)
            idx =1+idx
        self.counter+=1
        print self.rand[self.counter*self.batch]
        return Variable(inputd.cuda()),Variable(target.cuda())

class Prednet(nn.Module):
    def __init__(self, width_ratio=4, height_ratio=6, channels=3,batch=1,epoch = 10):
        super(Prednet, self).__init__()
        ## All the parameters below can be intialized in the __init__
        self.T = 20  # sequence length
        self.max_epoch = epoch
        self.lr = 1e-2
        self.batch_size = batch
        self.number_of_layers = 3
        self.R_size_list = [16,32,64] # channels of prediction of lstm
        self.Ahat_size_list = [channels,64,32] # channels of Ahat(l)
        self.params = None
        self.width_ratio = width_ratio # width=width_ratio*2**exp
        self.height_ratio = height_ratio # height=height_ratio*2**exp
        self.channels = channels
        self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = self.loss_fn.cuda()
        self.output_channels = channels*2
        self.exp = 7
        self.sensor_number = 7
        self.save_weights_interval = 1000
        self.initParams()
        self.optimizer = optim.SGD([
                weights for dic in self.params for weights in dic.values()
            ], lr=self.lr, momentum=0.9)
        self.dset = MyDataset(self.T,self.batch_size)
        self.norm32 = nn.BatchNorm2d(32)
        self.norm128 = nn.BatchNorm2d(128)
        self.norm16 = nn.BatchNorm2d(16)
        self.norm64 = nn.BatchNorm2d(64)
        self.count = 0

    # From pytorch LSTMCell
    def lstm(self,input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, padding=0):
        hx, cx = hidden
        gates = F.conv2d(input, w_ih, b_ih,padding=padding) + F.conv2d(hx, w_hh, b_hh, padding=padding)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy

    # Directly from the algorithm in the paper
    def forward(self ,input, error, state,):
        for l in reversed(range(0, self.number_of_layers)):
            p = self.params[l]
            error_projection = l == self.number_of_layers-1 and error[l]   \
                or torch.cat((error[l], F.upsample_nearest(state[l+1][0], scale_factor=2)), 1)
            state[l] = self.lstm(
                error_projection, state[l], p['lstm.weight_ih'], p['lstm.weight_hh'], p['lstm.bias_ih'], p['lstm.bias_hh'],padding=1
            )
        for l in range(0, self.number_of_layers):
            p = self.params[l]
            # Feeding Ahat of previous layer as input in this layer
            input_projection = l == 0 and input or F.relu(F.max_pool2d(F.conv2d(state_projection,p['convA.weight'],p['convA.bias'], padding=1), 2, 2))
            state_projection = l == 0 and F.hardtanh(F.conv2d(state[l][0],p['convAhat.weight'],p['convAhat.bias'],padding=1),0,1) \
                or F.relu(F.conv2d(state[l][0],p['convAhat.weight'],p['convAhat.bias'],padding=1))
            if l == 0 and self.count % self.T != 0 :
                buffer_image = torch.cat(
                    (input_projection.data[0].unsqueeze(0),
                        state_projection.data[0].unsqueeze(0)),0)
                # print buffer_image.size()
                # print input_projection.size()
                save_image(buffer_image,str(self.count%self.T)+'.png')
                self.count = self.count%self.T
            self.count += 1
            error[l] = F.relu(torch.cat((input_projection - state_projection, state_projection - input_projection),1))
        
        p = self.params[self.number_of_layers]
        # Sensor classifier is mounted over R3
        # print state[0][0].size()
        # print state[1][0].size()
        # print state[2][0].size()
        mp = nn.MaxPool2d(4, stride=4)
        maxp = mp(state[0][0])
        maxp = F.conv2d(maxp, p['preconv_weight'], p['preconv_bias'],stride = 1, padding=0)
        maxp = self.norm16(maxp)
        maxp = F.relu(maxp)
        conv = F.conv2d(maxp, p['conv_weight'], p['conv_bias'],stride = 2, padding=0)
        conv = self.norm32(conv)
        conv1 = F.conv2d(maxp, p['conv1_weight'], p['conv1_bias'],stride = 1, padding=0)
        conv1 = F.relu(self.norm128(conv1))
        conv2 = F.conv2d(conv1, p['conv2_weight'], p['conv2_bias'],stride = 2, padding=1)
        conv2 = F.relu(self.norm128(conv2))
        conv3 = F.conv2d(conv2, p['conv3_weight'], p['conv3_bias'],stride = 1, padding=0)
        conv3 = self.norm32(conv3)
        merge = F.relu(conv+conv3)
        avgpool = F.avg_pool2d(merge, kernel_size=2, stride=2)
        linear1 = F.tanh(F.linear(avgpool.view(avgpool.size(0), -1),p['linear_weight'],p['linear_bias']))
        linear2 = F.linear(linear1,p['linear2_weight'],p['linear2_bias'])
        # state[l] is a (hx,cx) tuple where hx is the output of the cell and cx is the hidden state of layer l
        return error, state, linear2

    def conv_init(self,no, ni, k): # k*k is the size of kernal
        return torch.Tensor(no, ni, k, k).normal_(0, 2/math.sqrt(ni*k*k)).cuda()

    def initParams(self):
        # try load the weights
        try: 
            f = open("prednet_weights", "r")
            self.params = pickle.load(f)
            print('Successfully load the weights!\n')
            f.close()
        except Exception:
            print('Fail to load the weights!\n')
            self.params = [None] * (self.number_of_layers+1)
            R_input_size_list = [x + 2*e for x,e in zip(self.R_size_list[1:] + [0],self.Ahat_size_list)]

            for l in range(0,self.number_of_layers):
                self.params[l] = {
                    # weights for lstm
                    'lstm.weight_ih':       self.conv_init(4*self.R_size_list[l], R_input_size_list[l], 3),
                    'lstm.bias_ih':         torch.zeros(4*self.R_size_list[l]).cuda(),
                    'lstm.weight_hh':       self.conv_init(4*self.R_size_list[l], self.R_size_list[l], 3),
                    'lstm.bias_hh':         torch.zeros(4*self.R_size_list[l]).cuda(),
                    # weights for R(l) --> Ahat(l)
                    'convAhat.weight':      self.conv_init(self.Ahat_size_list[l],self.R_size_list[l], 3),
                    'convAhat.bias':        torch.zeros(self.Ahat_size_list[l]).cuda(),
                          
                }

                if l > 0:
                    self.params[l]['convA.weight'] = self.conv_init(self.Ahat_size_list[l], self.Ahat_size_list[l-1], 3)
                    self.params[l]['convA.bias'] = torch.zeros(self.Ahat_size_list[l]).cuda()

                
            self.params[self.number_of_layers]={
            # weights for subsequent conv and fully connected layers
            'preconv_weight':     self.conv_init(self.R_size_list[0], self.R_size_list[0], 1),
            'preconv_bias':       torch.zeros(self.R_size_list[0]).cuda(),
            # feedforward stride = 2 
            'conv_weight':     self.conv_init(32, self.R_size_list[0], 1),
            'conv_bias':       torch.zeros(32).cuda(),
            # stride = 1
            'conv1_weight':     self.conv_init(128, self.R_size_list[0], 1),
            'conv1_bias':       torch.zeros(128).cuda(),
            # stride = 2
            'conv2_weight':     self.conv_init(128, 128, 3),
            'conv2_bias':       torch.zeros(128).cuda(),
            # stride = 1
            'conv3_weight':     self.conv_init(32, 128, 1),
            'conv3_bias':       torch.zeros(32).cuda(),

            'linear_weight':     torch.zeros(512,2048*2).cuda(),
            'linear_bias':       torch.zeros(512).cuda(),

            'linear2_weight':     torch.zeros(self.sensor_number,512).cuda(),
            'linear2_bias':       torch.zeros(self.sensor_number).cuda(),
            }
            for l in range(0,self.number_of_layers+1):
                for k,v in self.params[l].items():
                        self.params[l][k] = Variable(v, requires_grad=True)




    def train(self,):
        target_sequence = Variable(torch.zeros(net.T, net.batch_size, net.output_channels, net.height_ratio * 2 ** 7, net.width_ratio * 2 ** 7).cuda())
        input_sequence,target_parameter = self.dset.getbatch()
        # target_sequence = Variable(torch.zeros(input_sequence.size()))
        print('\n---------- Train a '+str(self.number_of_layers)+' layer network ----------')
        print('Input has size', input_sequence.size())
        print('Create a MSE criterion')

        print('Run for', self.max_epoch, 'Epoch')
        
        for epoch in range(0, self.max_epoch):
            self.dset.shuffle()
            #totloss = 0
            #print 30116/self.T/self.batch_size
	    f = open("prednet_weights", "w")
	    pickle.dump(self.params, f)
	    print("Save weights!")
	    f.close()

            for batch in range(0, 30116/self.T/self.batch_size-1):
                input_sequence,target_parameter = self.dset.getbatch()
                self.optimizer.zero_grad() # zero the gradient buffers
                loss = 0
                loss1 = 0
                loss2 = 0
                error = [Variable(torch.zeros(self.batch_size,2*self.Ahat_size_list[l],self.height_ratio*2 **(self.exp-l),self.width_ratio*2 **(self.exp-l)).cuda())
                    for l in range(0,self.number_of_layers)]
                state = [(
                    Variable(torch.zeros(self.batch_size, self.R_size_list[l], self.height_ratio*2 **(self.exp-l),self.width_ratio*2 **(self.exp-l)).cuda()),
                    Variable(torch.zeros(self.batch_size, self.R_size_list[l],self.height_ratio*2 **(self.exp-l), self.width_ratio*2 **(self.exp-l)).cuda())
                    ) for l in range(0,self.number_of_layers)]
                for t in range(0, self.T):
                    # repackage
                    for l in range(0,self.number_of_layers):
                        state[l] = (Variable(state[l][0].data),Variable(state[l][1].data))
                        error[l] = Variable(error[l].data)
                    error, state, predict_parameter = self.forward(input_sequence[t], error, state,)
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
                        loss1 += self.loss_fn(error[0], target_sequence[t])
                        loss2 += self.loss_fn(predict_parameter,target_parameter[t])
                
                print(' >>> Batch {:2d} image_loss: {:.3f} sensor_loss: {:.3f}'.format((batch + 1), loss1.data[0],loss2.data[0]))
                #totloss = totloss + loss
                loss = loss1+loss2
                loss.backward()
                self.optimizer.step()
                # if batch % 100 == 0:
                #     self.save_weight()
            # print(' > Epoch {:2d} loss: {:.3f}'.format((batch + 1), loss.data[0]))
            
            
    def save_weight(self,):
        f = open("prednet_weights", "w")
        pickle.dump(self.params, f)
        print("Save weights!")
        f.close()

if __name__ == '__main__':
    net = Prednet(
        2,1,3,
        batch = 5,
        epoch = 30
        )
    net = net.cuda()
    net.train()
    # model = torch.nn.DataParallel(net,device_ids=[0, 1, 2])
    # model.train()
