# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:53:06 2022
@author: fengshuai
"""
import argparse
from functools import partial
from pathlib import Path
from torch.autograd import Variable
from networks import get_deconf_net

import torch
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import losses

from datasets import get_transforms,get_dataloader,get_ood_transforms

from evaluation import compute_all_metrics
#from utils import get_ood_scores_MDS,get_ood_scores_odin,get_Mahalanobis_score,setup_logger,sample_estimator
from utils import *

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()  

gpu_id = 1  # gpu_id=0 or 1
torch.cuda.set_device(gpu_id)  

def main(args):
    output_path = Path(args.output_dir) / args.output_sub_dir  
    print('>>> Log dir: {}'.format(str(output_path)))
    output_path.mkdir(parents=True, exist_ok=True)
    
    # record console output
    setup_logger(str(output_path),args.method +'_'+str(args.temperature_odin)+'_'+str(args.epsilon_odin)+"_console.log")  
    
    benchmark = args.id
    if benchmark == "cifar10":
        num_classes = 10
    elif benchmark == "cifar100":
        num_classes = 100
    else:  # benchmark == "svhn"
        num_classes = 10
        
    ########----------1、Init Datasets ----------########
    print(">>>Initializing Datasets...")
    
    test_id_transform = get_transforms(name=benchmark,stage='test')
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
    test_id_loader = get_dataloader_default(name=benchmark, split='test',transform=test_id_transform)
    test_ood_loader_list = []
    for ood_ds in args.oods:
        test_ood_transform = get_ood_transforms(ID=benchmark, ood=ood_ds, stage='test')
        test_ood_loader = get_dataloader_default(name=ood_ds, split='test',transform=test_ood_transform)
        test_ood_loader_list.append(test_ood_loader)
    
    ########---------- 2、Load pre-trained Network model ----------########
    print(">>>Initializing Network...")
    print('>>> network: {}'.format(args.arch))
    print('>>> method: {}'.format(args.method))

    deconf_net = get_deconf_net(args.arch,num_classes)
    # load net classifier
    classifier_path = Path(args.classifier_path) # 
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path)) # 
        val_acc = cla_params['val_acc']
        deconf_net.load_state_dict(cla_params['state_dict'])  
        print('>>> load classifier from {} (classifiy acc {:.4f}%)'.format(str(classifier_path), val_acc))
        
    else:
        raise RuntimeError('<--- invalid net classifier path: {}'.format(classifier_path))
        
    deconf_net.eval()
    if torch.cuda.is_available():
        if args.ngpu > 1:
            deconf_net = torch.nn.DataParallel(deconf_net, device_ids=list(range(args.ngpu)))
        if args.ngpu > 0:
            deconf_net.cuda()
            # torch.cuda.manual_seed(1)
    cudnn.benchmark = True
    
    ########---------- 3、OOD Detection ----------########
    result_dic_list = [] # various metrics store
    
    # 3.1 OOD scores
    if args.method == 'msp':
        id_score, right_score, wrong_score = get_msp_score(test_id_loader,deconf_net, in_dist=True)
    elif args.method == 'energy':
        id_score, right_score, wrong_score = get_energy_score(test_id_loader,deconf_net, args.temperature_energy,in_dist=True)
    elif args.method == "odin":
        id_score,right_score, wrong_score = get_odin_score(test_id_loader,deconf_net,args.temperature_odin, args.epsilon_odin, in_dist=True)
    elif args.method == "M":
        _,right_score, wrong_score = get_msp_score(test_id_loader, deconf_net, in_dist=True) 
        train_loader = get_dataloader_default(name=benchmark,split='train',transform=test_id_transform) 
        # set information about feature extaction
        temp_x = torch.rand(2,3,32,32)  # torch.Size([2, 3, 32, 32])
        temp_x = Variable(temp_x)
        temp_x = temp_x.cuda()
        
        temp_list = net.feature_list(temp_x)[1]  
        num_output = len(temp_list)  
        feature_list = np.empty(num_output)  # feature_list =[128.]
        count = 0
        for out in temp_list: # out=torch.Size([2, 128, 8, 8])
            feature_list[count] = out.size(1)  
            count += 1
        
        print('get sample mean and covariance',count)
        sample_mean,precision = sample_estimator(net, num_classes, feature_list, train_loader)
        id_score = get_Mahalanobis_score(net, test_id_loader, num_classes, sample_mean, precision, count-1, args.magnitude)
    else:
         raise RuntimeError('<--- invalid method name: {}'.format(args.method))
    
    id_labels = np.zeros(len(id_score)) # id_labels all 0
    
    num_right = len(right_score)
    num_wrong = len(wrong_score)
    id_test_error = 100 * num_wrong / (num_wrong + num_right)
    print('{}_id_test_error rate : {:.2f}'.format(args.method,id_test_error))
   
    # 3.2 OOD test
    for test_out_loader in test_ood_loader_list:
        result_dic = {'name': test_out_loader.dataset.name}
        
        if args.method =='msp':
            out_score = get_msp_score(test_out_loader,deconf_net,in_dist=False)
        elif args.method == 'energy':
            out_score = get_energy_score(test_out_loader,deconf_net,args.temperature_energy,in_dist=False)
        elif args.method == "odin":
            out_score = get_odin_score(test_out_loader,deconf_net,args.temperature_odin, args.epsilon_odin,in_dist=False)
        elif args.method =="M": 
            out_score = get_Mahalanobis_score(net, test_out_loader, num_classes, sample_mean, precision, count-1, args.magnitude)
        else: 
            raise RuntimeError('<--- invalid method name: {}'.format(args.method))
            
        out_labels = np.ones(len(out_score)) 
        
        # scores and lables
        scores = np.concatenate([id_score, out_score]) 
        labels = np.concatenate([id_labels, out_labels])
        
        result_dic['fpr_at_tpr'], result_dic['auroc'], result_dic['aupr_in'], result_dic['aupr_out'] = compute_all_metrics(scores, labels, verbose=False)
        result_dic['id_test_error'] = id_test_error
        result_dic_list.append(result_dic)
        
        print('---> [ID: {:7s} - OOD: {:9s}] [auroc: {:3.3f}%, aupr_in: {:3.3f}%, aupr_out: {:3.3f}%, fpr@95tpr: {:3.3f}%]'.format(
            test_id_loader.dataset.name, test_out_loader.dataset.name, 100. * result_dic['auroc'], 100. * result_dic['aupr_in'], 100. * result_dic['aupr_out'], 100. * result_dic['fpr_at_tpr']))
        
    # save results
    result = pd.DataFrame(result_dic_list) # pandas.DataFrame 是一个表格型的数据结构，
    log_path = output_path / (args.method +'_'+str(args.temperature_odin)+'_'+str(args.epsilon_odin)+'.csv')
    result.to_csv(str(log_path), index=False, header=True) # header=True，列名的别名; index=True,写行名(索引)
    # index=0:不保存行索引。 header=0 不保存列名。



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect ood')
    
     # file saving
    parser.add_argument('--output_dir', help='dir to store log', default='OOD_test_compact_loss+disper_loss_cifar100_seed_4')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='metrics_results')
    parser.add_argument('--classifier_path', type=str, default='cifar100_resnet34/compact_loss+disper_loss_lr_0.1_decay_0.0005_bsz_256_epochs_300_seed_4/cifar100_300.pth') 
    # datasets
    parser.add_argument('--data_dir', help='directory to store datasets', default='/root/iip_datasets')
    parser.add_argument('--id', type=str, default='cifar100')
    parser.add_argument('--oods', nargs='+', default=['cifar10', 'tinc', 'tinr', 'lsunc', 'lsunr', 'isun','svhn','dtd', 'places365_10k'])  # default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365_10k', 'isun']
    # network architecture and path
    parser.add_argument('--arch', type=str, default='resnet34')
    # Optimization options：
    parser.add_argument('--method', type=str, default='odin')
    parser.add_argument('--batch_size', type=int, default=200) # 128
    # arguments for ODIN
    parser.add_argument('--temperature_odin', type=float, default=10.0) #10.0
    parser.add_argument('--epsilon_odin', type=float, default=0.005) # 0.005
    # arguments for MDS
    # parser.add_argument('--epsilon_mds', type=float, default=0.0005)  # 
    # arguments for energy
    parser.add_argument('--temperature_energy', type=int, default=1.0)
    # Acceleration
    parser.add_argument("--ngpu",type=int,default=1,help="number of GPUs to use,0=cpu")
    parser.add_argument("--prefetch", type=int,default=4,help="pre_fetching threads.")

    args = parser.parse_args() # 调用parse_args()方法进行解析参数，即读取命令行输入参数
    
    main(args)
    
