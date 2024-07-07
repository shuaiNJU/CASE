import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from .metrics import compute_all_metrics

device = 1
torch.cuda.set_device(device)
class Evaluator():
    def __init__(self, deconf_net,compact_loss,disper_loss):
        self.deconf_net = deconf_net
        ##########################################
        self.compact_loss = compact_loss 
        self.disper_loss = disper_loss
        ##########################################
    
    def eval_classification(self, data_loader: DataLoader):
        self.deconf_net.eval()
        
        total = 0
        correct = 0 
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                data,target = batch
                data,target = data.cuda(),target.cuda()

                # forward
                features,logits,quotients,class_centroid = self.deconf_net(data) # 3 outpus：features, logits=-distance，h/g
                
                # two loss
                c_loss = self.compact_loss(quotients,target) 
                d_loss= self.disper_loss(class_centroid)
                
                loss = c_loss+d_loss
                total_loss += loss.item()
                
                # accuracy
                pred = quotients.data.max(dim=1)[1]
                total += target.size(0)
                correct += pred.eq(target.data).sum().item()
                
        # average on sample
        print("-validation: [loss:{:.4f} | accuracy:{:.4f}%]".format(
            total_loss/len(data_loader),
            100.*correct/total
            )
        )
        
        metrics ={
            'val_loss':total_loss/len(data_loader),
            'val_acc': 100.*correct/total
            }      
        return metrics
    
   
    
    