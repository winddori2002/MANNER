import warnings
warnings.filterwarnings('ignore')

import os
import json
import yaml
from torch.utils.data import Dataset, DataLoader 

from src.dataset import *
from src.utils import *
from src.evaluation import *
from src.train import *
from config import *

def main():
    
    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))

    seed_init()

    if args.action == 'train':
        
        kwargs = {"matching": args.dataset['matching'], "sample_rate":16000}
        length = int(args.setting['segment'] * args.setting['sample_rate'])
        stride = int(args.setting['stride'] * args.setting['sample_rate'])

        train_dataset = TrainDataset(args.dataset['train'], length=length, stride=stride, valid=args.dataset['val'], pad=args.setting['pad'], **kwargs)
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)   
        val_dataset   = ValDataset(args.dataset['train'], valid=args.dataset['val'], **kwargs)
        val_loader    = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_worker)
        data_loader   = {'train':train_loader, 'val':val_loader}
        
        trainer = Trainer(data_loader, args)
        trainer.train()
        
        tester = Tester(args)
        print('---Test score---')
        tester.test()   

    else:
        tester = Tester(args)
        print('---Test score---')
        tester.test()

if __name__ == "__main__":
    main()
