from collections import deque
import random
import torchvision
import torchvision.transforms as T
#import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader,Dataset
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pickle

# lass TensorDataset(Dataset):
#     """Dataset wrapping data and target tensors.

#     Each sample will be retrieved by indexing both tensors along the first
#     dimension.

#     Arguments:
#         data_tensor (Tensor): contains sample data.
#         target_tensor (Tensor): contains sample targets (labels).
#     """

#     def __init__(self, data_tensor, target_tensor):
#         assert data_tensor.size(0) == target_tensor.size(0)
#         self.data_tensor = data_tensor
#         self.target_tensor = target_tensor

#     def __getitem__(self, index):
#         return self.data_tensor[index], self.target_tensor[index]

#     def __len__(self):
#         return self.data_tensor.size(0)
# http://pytorch.org/docs/data.html?highlight=dataloader#torch.utils.data.DataLoader
trans=T.Compose([
        	T.Scale(128),T.CenterCrop(128), T.ToTensor()
                             ])
max = 10000
length = 10
batch_size = 2
toPIL = T.Compose([T.ToPILImage()])

def transform(num):
	filename = '/media/HDD1/Datasets/GTA/2/'+str(num)+ '.png'
	im = Image.open(filename)
	return trans(im)#.unsqueeze(0)

def img_show(img):
        im = toPIL(img)
        im.show()

def save(tensor,filename):
    	save_image(tensor, filename, nrow=8, padding=2)

class MyDataset(Dataset):
    def __init__(self,len = 10,max=10000,batch = 10):
        self.length = len
        self.max = max
        self.batch = batch
        self.fname = '/media/HDD1/Datasets/GTA/2/dataset.txt'
	self.target = np.loadtxt(self.fname)
	self.targetdata = torch.from_numpy(self.target)
	self.targetdata = self.targetdata[0:max,1:8]

    def get(self, idx):
    	traindata = torch.zeros(self.length,3,128,128)
    	index = 0
    	for i in range(idx,self.length+idx):
		img = transform(i)
		img = img[0:3]
		traindata[index-1] = img
		index +=1
	target = self.targetdata[idx:idx+length]
	return traindata,target

    def getbatch(self):
    	rand = np.random.randint(0,self.max,size = (self.batch))
    	inputd = torch.zeros(self.length,self.batch,3,128,128)
    	target = torch.zeros(self.length,self.batch,7)
    	idx = 0
    	for i in rand:
    		inputd[:,idx,:,:,:],target[:,idx,:] = self.get(i)
    		idx =1+idx
    	return inputd,target

def main():
	dset = MyDataset()
	i,o = dset.getbatch()
	print i,o
	# traindata = torch.zeros(max,3,128,128)
	# fname = '/media/HDD1/Datasets/GTA/2/dataset.txt'
	# target = np.loadtxt(fname)
	# targetdata = torch.from_numpy(target)
	# targetdata = targetdata[0:max,1:8]
	# for i in xrange(1,max+1):
	# 	img = transform(i)
	# 	img = img[0:3]
	# 	traindata[i-1] = img

	# trainset = TensorDataset(traindata,targetdata)
	# f = open("trainset", "w")
	# pickle.dump(trainset, f)
 #    	f.close()
	# # trainloader = DataLoader(trainset, batch_size=batch_size,)

	# # dataiter = iter(trainloader)
	# # input, target = dataiter.next()
if __name__ == "__main__":
	main()
