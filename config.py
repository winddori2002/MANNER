import os 
import argparse

def get_config():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    
    # dataset
    parser.add_argument('--train', type=str, default='data_path/train', help='Train path')
    parser.add_argument('--test', type=str, default='data_path/test', help='Test path')
    parser.add_argument('--val', type=list, default=[282,287], help='Valid speaker')
    parser.add_argument('--matching', type=str, default='sort', help='Matching')

    #setting
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--segment', type=int, default=4, help='Segment') # segment signal per 4 seconds
    parser.add_argument('--pad', type=bool, default=True, help='Pad')
    parser.add_argument('--set_stride', type=int, default=1, help='Stride') # segment signal with overlapped 1 second

    #manner
    parser.add_argument('--in_channels', type=int, default=1, help='In channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Out channels')
    parser.add_argument('--hidden', type=int, default=60, help='Hidden')
    parser.add_argument('--depth', type=int, default=4, help='Depth')
    parser.add_argument('--kernel_size', type=int, default=8, help='Kernel size')
    parser.add_argument('--stride', type=int, default=4, help='Stride')
    parser.add_argument('--growth', type=int, default=2, help='Growth') # channel growth rate 
    parser.add_argument('--head', type=int, default=1, help='Head') # number of heads in global attention
    parser.add_argument('--segment_len', type=int, default=64, help='Segment len') # chunk size to split the sigal
    
    #basic 
    parser.add_argument('--save_enhanced', type=bool, default=False, help='Save option') # save the enhanced speech in test phase
    parser.add_argument('--enhanced_path', type=str, default='./enhanced', help='Enhanced path') # save the enhanced speech in test phase
    parser.add_argument('--model_path', type=str, default='./weights/', help='Model path')
    parser.add_argument('--model_name', type=str, default='manner_base.pth', help='Model name') # select manner_ {small, base, large}
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=350, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--loss', type=str, default='l1', help='Loss') # {l1:L1 loss, ch:chabonnier loss}
    parser.add_argument('--stft_loss', type=bool, default=True, help='Stft loss') # always apply stft loss, if you don't want to use, change the default as False
    parser.add_argument('--stft_sc_factor', type=float, default=0.5, help='Stft sc factor')
    parser.add_argument('--stft_mag_factor', type=float, default=0.5, help='Stft mag factor')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Checkpoint') # If you want to train with pre-trained, or resume set True
    parser.add_argument('--aug', type=bool, default=False, help='Augmentation')
    parser.add_argument('--aug_type', type=str, default='tempo', help='Augmentation type')  # {tempo, speed, shift}

    # device 
    parser.add_argument('--device', type=str, default='cuda:0', help='Gpu device')
    parser.add_argument('--env', type=str, default='local', help='Enviornment')
    parser.add_argument('--num_worker', type=int, default=0, help='Num workers')

    # logging setting
    parser.add_argument('--logging', type=bool, default=False, help='Logging')
    parser.add_argument('--logging_cut', type=int, default=-1, help='Logging cut') # logging after the epoch of logging_cut
    
    arguments = parser.parse_args()
    
    return arguments
