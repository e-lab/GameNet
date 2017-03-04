import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn as nn
import pickle
import torch.optim as optim

torch.manual_seed(0)

USE_CUDA = torch.cuda.is_available()

class Variable(Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

class Prednet(nn.Module):
    def __init__(self, width_ratio=4, height_ratio=6, channels=3):
        super(Prednet, self).__init__()
        ## All the parameters below can be intialized in the __init__
        self.T = 10  # sequence length
        self.max_epoch = 10
        self.lr = 1e-2
        self.batch_size = 1
        self.number_of_layers = 3
        self.R_size_list = [16,32,64] # channels of prediction of lstm
        self.Ahat_size_list = [channels,64,32] # channels of Ahat(l)
        self.params = [None] * (self.number_of_layers+1)
        self.width_ratio = width_ratio # width=width_ratio*2**exp
        self.height_ratio = height_ratio # height=height_ratio*2**exp
        self.channels = channels
        self.loss_fn = torch.nn.MSELoss()
        self.output_channels = channels*2
        self.exp = 7
        self.sensor_number = 6
        self.save_weights_interval = 5
        self.initParams()
        self.optimizer = optim.SGD([
                weights for dic in self.params for weights in dic.values()
            ], lr=self.lr, momentum=0.9)

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
            
            error[l] = F.relu(torch.cat((input_projection - state_projection, state_projection - input_projection),1))
        
        p = self.params[self.number_of_layers]
        # Sensor classifier is mounted over R3
        conv = F.conv2d(state[2][0], p['conv_weight'], p['conv_bias'],stride = 2, padding=0)
        conv1 = F.conv2d(state[2][0], p['conv1_weight'], p['conv1_bias'],stride = 1, padding=0)
        conv2 = F.conv2d(conv1, p['conv2_weight'], p['conv2_bias'],stride = 2, padding=1)
        conv3 = F.conv2d(conv2, p['conv3_weight'], p['conv3_bias'],stride = 1, padding=0)
        merge = conv+conv3
        avgpool = F.avg_pool2d(merge, kernel_size=2, stride=2)
        linear1 = F.tanh(F.linear(avgpool.view(avgpool.size(0), -1),p['linear_weight'],p['linear_bias']))
        linear2 = F.tanh(F.linear(linear1,p['linear2_weight'],p['linear2_bias']))
        # state[l] is a (hx,cx) tuple where hx is the output of the cell and cx is the hidden state of layer l
        return error, state, linear2

    def conv_init(self,no, ni, k): # k*k is the size of kernal
        return torch.Tensor(no, ni, k, k).normal_(0, 2/math.sqrt(ni*k*k))

    def initParams(self):
        # try load the weights
        try: 
            f = open("prednet_weights", "r")
            self.params = pickle.load(f)
            print('Successfully load the weights!\n')
        except Exception:
            print('Fail to load the weights!\n')
            
            R_input_size_list = [x + 2*e for x,e in zip(self.R_size_list[1:] + [0],self.Ahat_size_list)]

            for l in range(0,self.number_of_layers):
                self.params[l] = {
                    # weights for lstm
                    'lstm.weight_ih':       self.conv_init(4*self.R_size_list[l], R_input_size_list[l], 3),
                    'lstm.bias_ih':         torch.zeros(4*self.R_size_list[l]),
                    'lstm.weight_hh':       self.conv_init(4*self.R_size_list[l], self.R_size_list[l], 3),
                    'lstm.bias_hh':         torch.zeros(4*self.R_size_list[l]),
                    # weights for R(l) --> Ahat(l)
                    'convAhat.weight':      self.conv_init(self.Ahat_size_list[l],self.R_size_list[l], 3),
                    'convAhat.bias':        torch.zeros(self.Ahat_size_list[l]),
                          
                }

                if l > 0:
                    self.params[l]['convA.weight'] = self.conv_init(self.Ahat_size_list[l], self.Ahat_size_list[l-1], 3)
                    self.params[l]['convA.bias'] = torch.zeros(self.Ahat_size_list[l])

                
            self.params[self.number_of_layers]={
            # weights for subsequent conv and fully connected layers 
            # feedforward stride = 2 
            'conv_weight':     self.conv_init(32, 2*self.Ahat_size_list[-1], 1),
            'conv_bias':       torch.zeros(32),
            # stride = 1
            'conv1_weight':     self.conv_init(128, 2*self.Ahat_size_list[-1], 1),
            'conv1_bias':       torch.zeros(128),
            # stride = 2
            'conv2_weight':     self.conv_init(128, 128, 3),
            'conv2_bias':       torch.zeros(128),
            # stride = 1
            'conv3_weight':     self.conv_init(32, 128, 1),
            'conv3_bias':       torch.zeros(32),

            'linear_weight':     torch.zeros(512,2048),
            'linear_bias':       torch.zeros(512),

            'linear2_weight':     torch.zeros(self.sensor_number,512),
            'linear2_bias':       torch.zeros(self.sensor_number),
            }

            for l in range(0,self.number_of_layers+1):
                for k,v in self.params[l].items():
                        self.params[l][k] = Variable(v, requires_grad=True)



    def train(self,input_sequence,target_sequence,target_parameter):
        print('\n---------- Train a '+str(self.number_of_layers)+' layer network ----------')
        print('Input has size', list(input_sequence.data.size()))
        print('Create a MSE criterion')

        print('Run for', self.max_epoch, 'iterations')
        for epoch in range(0, self.max_epoch):
            self.optimizer.zero_grad() # zero the gradient buffers
            loss = 0
            error = [Variable(torch.zeros(self.batch_size,2*self.Ahat_size_list[l],self.width_ratio*2 **(self.exp-l),self.height_ratio*2 **(self.exp-l)))
                for l in range(0,self.number_of_layers)]
            state = [(
                Variable(torch.zeros(self.batch_size, self.R_size_list[l], self.width_ratio*2 **(self.exp-l),self.height_ratio*2 **(self.exp-l))),
                Variable(torch.zeros(self.batch_size, self.R_size_list[l], self.width_ratio*2 **(self.exp-l),self.height_ratio*2 **(self.exp-l)))
                ) for l in range(0,self.number_of_layers)]
            
            for t in range(0, self.T):
                error, state, predict_parameter = self.forward(input_sequence[t], error, state,)
               
                loss += self.loss_fn(error[0], target_sequence[t])
                loss += self.loss_fn(predict_parameter,target_parameter[t])
            print state[2][0].data.size()

            print(' > Epoch {:2d} loss: {:.3f}'.format((epoch + 1), loss.data[0]))

            loss.backward()
            self.optimizer.step()
            
            if epoch % self.save_weights_interval == 0:
                f = open("prednet_weights", "w")
                pickle.dump(self.params, f)
                f.close()


if __name__ == '__main__':
    net = Prednet(1,1,3)
    input_sequence = Variable(torch.rand(net.T, net.batch_size, net.channels, net.width_ratio * 2**7, net.height_ratio * 2**7))
    target_sequence = Variable(torch.zeros(net.T, net.batch_size, net.output_channels, net.width_ratio * 2 ** 7, net.height_ratio * 2 ** 7))
    target_parameter = Variable(torch.zeros(net.T,net.batch_size,6))
    net.train(input_sequence,target_sequence,target_parameter)
