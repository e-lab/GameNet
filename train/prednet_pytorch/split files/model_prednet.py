import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn as nn
import pickle
from torchvision.utils import save_image

class Prednet(nn.Module):
    def __init__(self, width_ratio=4, height_ratio=6, channels=3, batch=20):
        super(Prednet, self).__init__()
        ## All the parameters below can be intialized in the __init__
        self.T = 20  # sequence length
        self.batch_size = batch
        self.number_of_layers = 3
        self.R_size_list = [16,32,64] # channels of prediction of lstm
        self.Ahat_size_list = [channels,64,32] # channels of Ahat(l)
        self.params = None
        self.width_ratio = width_ratio # width=width_ratio*2**exp
        self.height_ratio = height_ratio # height=height_ratio*2**exp
        self.channels = channels
        self.loss_fn = torch.nn.MSELoss().cuda()
        self.output_channels = channels*2
        self.exp = 7
        self.sensor_number = 7
        self.save_weights_interval = 1000
        self.initParams()
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
            # Feeding Ahat of previous error as input in this layer
            input_projection = l == 0 and input or F.relu(F.max_pool2d(F.conv2d(error[l-1],p['convA.weight'],p['convA.bias'], padding=1), 2, 2))
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
                    self.params[l]['convA.weight'] = self.conv_init(self.Ahat_size_list[l], 2*self.Ahat_size_list[l-1], 3)
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
