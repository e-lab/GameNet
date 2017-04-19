from argparse import ArgumentParser


def get_args():
    """
    Parse input arguments for the training script
    :return: all the input arguments
    """
    parser = ArgumentParser(description='e-Lab Gesture Recognition Script')
    _ = parser.add_argument
    _('--datadir',  type=str,   default='/home/elab/Datasets/GTAV/2/', help='dataset location')
    _('--savedir',  type=str,   default='/home/elab/Datasets/GTAV/2/', help='folder to save outputs')
    _('--model',    type=str,   default='models/model.py')
    _('--channels', type=int,   default=3)
    _('--height',   type=int,   default=128)
    _('--width',    type=int,   default=256)
    _('--sensors',  type=int,   default=7)
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
    args = parser.parse_args()
    return args
