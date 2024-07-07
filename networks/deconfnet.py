# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

############################################################################
class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EuclideanDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
       # use pedcc to replace w
       #  self.h.weight.data = tensor_pedcc  #self.h.weight.data torch.Size([10, 512]) 

    def forward(self, x):
        # class_centroid = self.h.weight.data # self.h.weight.data torch.Size([100, 512])
        class_centroid = self.h.weight
        
        # print('class_centroid.shape=',class_centroid.shape)   
        distances = torch.cdist(F.normalize(x), F.normalize(class_centroid), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        # distance.shape=torch.Size([128,10 or 100])
        
        distances_square = torch.pow(distances,2)
        logits = -distances_square
        
        return logits,class_centroid
###########################################################################
def get_h(in_features,num_classes):
    # inner, euclidean, cosine
    h = EuclideanDeconf(in_features,num_classes)
    return h
##################################################################     
# h(x)/g(x)
class DeconfNet(nn.Module):
    def __init__(self, model,in_features,h):
        super(DeconfNet, self).__init__()

        self.model = model
        self.h =h
        self.g = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
            )
        #self.softmax = nn.Softmax()
    
    def forward(self, x):
        features= self.model(x) 
        # logits = self.h(features)   # -distance
        ##################################
        # features = F.normalize(features)  
        #################################
        logits,class_centroid = self.h(features)  # 返回两个值，一个是logits = -distances_square，一个是class_centroid
        denominators = self.g(features)  # torch.Size([128, 1])
        #print('denominators',denominators)
        # Now, broadcast the denominators per image across the numerators by division
        quotients = logits / denominators
        
        # logits, numerators, and denominators
        return  features,logits,quotients,class_centroid