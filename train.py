# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:14:48 2022
@author: fengshuai
"""
import os
import random
import torch
import torch.backends.cudnn as cudnn
import argparse 
import numpy as np
import time
import copy
import losses
from pathlib import Path  
from functools import partial

from utils import setup_logger
import torch.optim as optim
from datasets import get_transforms,get_dataloader
from networks import get_deconf_net
from trainers import get_trainer
from evaluation import Evaluator

device = 1
torch.cuda.set_device(device)

def main(args): 

    # model saving path
    args.model_name = '{}_lr_{}_decay_{}_bsz_{}_epochs_{}_seed_{}'.\
        format(args.loss_name,args.lr, args.weight_decay,
               args.batch_size,args.epochs,args.seed)
    args.model_path = './{}_{}'.format(args.dataset,args.arch)
    args.save_folder = os.path.join(args.model_path, args.model_name)
    # store net and console log by training method
    save_folder = Path(args.save_folder)    
    print(">>> save_folder: {}".format(str(save_folder)))
    save_folder.mkdir(parents=True, exist_ok=True)
   
    # record console output
    setup_logger(str(save_folder),"console.log") 

    benchmark = args.dataset
    if benchmark == "cifar10":
        num_classes = 10
    elif benchmark == "cifar100":
        num_classes = 100
    else: # benchmark == "svhn"
        num_classes = 10
        
    ########----------1、Init Datasets ----------########
    # 1.1、get dataset transform
    train_transform = get_transforms(benchmark, stage='train')
    val_transform = get_transforms(benchmark, stage='test')
    print('>>> Dataset: {}'.format(benchmark))
    
    # 1.2 get dataloader
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        name=benchmark,
        batch_size=args.batch_size,
        num_workers=args.prefetch
    )
    
    train_loader = get_dataloader_default(
        split='train',
        transform=train_transform,
        shuffle=True
    )

    test_loader = get_dataloader_default(
        split='test',
        transform=val_transform,
        shuffle=False
    )
    
    ########----------2、Init Networks ----------########
    print(">>>Initializing Network...")
    print('>>> network: {}'.format(args.arch))
    #####################################################
    # about network output
    #embedding_prototype_dis = losses.Embedding_prototype_distance  
    # about two loss
    compact_loss = losses.Compactness_loss()  
    disper_loss = losses.Dispersion_loss()
    
    #####################################################
    #net = get_network(args.arch,num_classes,embedding_prototype_dis)
    deconf_net = get_deconf_net(args.arch,num_classes)
    
    # move net to gpu device
    if torch.cuda.is_available():
        if args.ngpu > 1:
            deconf_net = torch.nn.DataParallel(deconf_net, device_ids=list(range(args.ngpu)))
        if args.ngpu > 0:
            deconf_net.cuda()
            # torch.cuda.manual_seed(1)
    
    cudnn.benchmark = True  # fire on all cylinders 
    
    parameters = []
    h_parameters = []
    for name, parameter in deconf_net.named_parameters():
        # print('name=',name)
        if name == 'h.h.weight' or name == 'h.h.bias':
            h_parameters.append(parameter)
        else:
            parameters.append(parameter)

    ########---------- 3、Init Trainer ----------##########
    print('>>> Optimizer: SGD  | Scheduler: LambdaLR')
    print('>>> epochs: {:3d} | Lr: {:.5f} | Weight_decay: {:.5f} | Momentum: {:.2f}| batch_size: {:3d} | arch: {}'.format(args.epochs, args.lr, args.weight_decay, args.momentum,args.batch_size,args.arch))
    print('save_folder=',args.save_folder)
    # 3.1 optimizer
    optimizer = optim.SGD(parameters, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma = 0.1)
    
    h_optimizer = optim.SGD(h_parameters, lr = args.lr, momentum = args.momentum) # No weight decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma = 0.1)
    
    trainer = get_trainer(deconf_net,train_loader,optimizer,h_optimizer,scheduler,h_scheduler,compact_loss,disper_loss) # 注意这里传入的是deconf_net
    
    #########---------- 4、Start Training ----------########
    print('>>>>Beginning Training\n')
    
    evaluator = Evaluator(deconf_net,compact_loss,disper_loss)  # evaluator是根据类Evaluator创建的实例
    begin_time = time.time()
    best_val_acc = 0.0
    start_epoch = 1
    
    # main loop
    for epoch in range(start_epoch,args.epochs+1):
        print("---> Epoch: ",epoch)
        
        trainer.train_epoch() # train
        val_metrics = evaluator.eval_classification(test_loader) # validation
        
        # save best result
        val_best = val_metrics['val_acc'] > best_val_acc  # True or False
        best_val_acc = max(val_metrics['val_acc'], best_val_acc)
        
        if val_best:
            val_best_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(deconf_net.state_dict()),
                'val_acc': best_val_acc
            }
        
        # save the  model every 50 epoch
        if epoch % args.epochs ==0:
            last_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': deconf_net.state_dict(),
                'val_acc': val_metrics['val_acc']
            }
            save_path=args.dataset+'_'+str(epoch)+'.pth'
            last_path = save_folder / save_path
            torch.save(last_state, str(last_path))
        # save the last epoch model
        if epoch == args.epochs:
            last_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': deconf_net.state_dict(),
                'val_acc': val_metrics['val_acc']
            }
            
        print("-Time:{:5d}s".format(int(time.time()-begin_time)),flush=True)
        print("------------epoch-{}-is over-------------".format(epoch))
    
    print('---> Best classify acc: {:.4f}%, epoch:{:2d}'.format(val_best_state['val_acc'], val_best_state['epoch']))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Classifier')  
    # datasets
    parser.add_argument('--data_dir', help='directory to store datasets', default='/root/iip_datasets')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--seed', type=int, default=4, help='seed=4 for CIFAR100 reproduction')
    # network architecturecl
    parser.add_argument('--arch', type=str, default='resnet34')
    # Optimization options：c
    parser.add_argument('--loss_name', type=str, default='compact_loss+disper_loss') # disper_loss
    parser.add_argument('--lr', type=float, default=0.1,help='The initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    # Acceleration
    parser.add_argument("--ngpu",type=int,default=1,help="number of GPUs to use,0=cpu")
    parser.add_argument("--prefetch", type=int,default=10,help="pre_fetching threads.")  # 10块cpu
    
    args = parser.parse_args() 
    # setting seed
    setup_seed(args.seed)
    main(args)