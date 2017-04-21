# Prednet for self-driving
## Dependencies

 + [*pytorch*](http://pytorch.org)

```
# for python 3.5 and cuda 8.0
pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl 
pip install torchvision
```

### Options
```
    parser = ArgumentParser(description='e-Lab Gesture Recognition Script')
    _ = parser.add_argument
    _('--datadir',  type=str,   default='/home/elab/Datasets/GTAV/2/', help='dataset location')
    _('--savedir',  type=str,   default='/home/elab/Datasets/GTAV/2/', help='folder to save outputs')
    _('--model',    type=str,   default='models/model.py')
    _('--fileNum',  type=int,   default=30116)
    _('--batchSize',type=int,   default=16)
    _('--seqLen',   type=int,   default=10)
    _('--dim',      type=int,   default=(256, 144), nargs=2, help='input image dimension as tuple (HxW)', metavar=('W', 'H'))
    _('--lr',       type=float, default=1e-2, help='learning rate')
    _('--eta',      type=float, default=0.9, help='momentum')
    _('--seed',     type=int,   default=1, help='seed for random number generator')
    _('--epochs',   type=int,   default=30, help='# of epochs you want to run')
    _('--devID',    type=int,   default=0, help='GPU ID to be used')
    _('--cuda',     type=bool,  default=True, help='use CUDA')
```

```
python3 main.py 
```

#### media

-Sample prednet image and video


## January

+ Set up environment for data collecting

## Feburary

+ Set up environment for reinforcement learning

+ Finish  tcp server communication script to get feedback from environment

+ Collect Dataset (60k images and sets of sensor readings)

+ Finish basic structure for reinforcement learning (waiting for embedding of prednet) 

## March

+ Finish LSTM module

+ Finish Prednet class

+ Finish class for image transformation and batch getting

+ Add head for sensor output to prednet

+ Modify Prednet (Feed prediction A_hat to subsequent layers instead of error)

## Apirl

+ Start training (length = 4, batch = 64)

+ Normalize sensor reading

+ Finish training for original and modified prednet with (20k batches)

* input
![](https://github.com/e-lab/GameNet/blob/master/train/prednet_pytorch/media/input.png)

* prediction
![](https://github.com/e-lab/GameNet/blob/master/train/prednet_pytorch/media/preview.png)

+ Problem identified: Sensor fitting is not ideal.



