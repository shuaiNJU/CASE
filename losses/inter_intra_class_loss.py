import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle

# latent_variable_dim = 512
class_num = 100  # The number of class centroids
class Compactness_loss(nn.Module):
    def __init__(self): #10.0
        super(Compactness_loss, self).__init__()
        
    def forward(self, quotients, targets):
        c_loss = F.cross_entropy(quotients, targets)
        return c_loss
    
class Dispersion_loss(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self): 
        super(Dispersion_loss, self).__init__()
        # self.temp = temp
    def forward(self, class_centroid):# 
        
        num_classes = class_centroid.size(0)
        ####################################
        class_centroid_normalized = F.normalize(class_centroid)
        ########################################
        # obatain |wi-wj|
        distances = torch.cdist(class_centroid_normalized, class_centroid_normalized, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") # 对角线已经为0了
        distances = torch.pow(distances,2) #

        distances = torch.exp(-distances).cuda()  # e^(wi*wj)
        exp_distances = torch.sum(distances,dim=1)/(num_classes-1)
        d_loss = torch.mean(exp_distances)

        return d_loss
