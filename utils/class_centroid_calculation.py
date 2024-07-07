import torch
import numpy as np
import torch.nn.functional as F

latent_variable_dim = 512

############################################################################################################
# 遍历所有batch,使用移动指数平均获得每个类的centroid
def get_class_centroid(net,train_loader,num_classes):
    # 初始化类质心
    class_centroid = torch.randn(num_classes,latent_variable_dim) # (0,1)正太分布
    class_centroid = F.normalize(class_centroid,dim=1).cuda() # 标准化初始类质心,并放到gpu上
    print('调用一次初始化的类质心')
    for sample in train_loader:
        data, labels = sample
        data, labels = data.cuda(), labels.cuda()
           
        # forward
        feature,_= net(data) # out1和out2分别输出的feature和logits=-distance
        features =feature.detach() # .detahc() 与计算图分离，得到的这个tensor永远不需要计算其梯度
        with torch.no_grad():
            labels_t2n = labels.detach().cpu().numpy() # numpy.ndarray
            # update clss centroids with a weighted moving average in each batch
            for one_label in torch.unique(labels): # torch.unique()的功能类似于数学中的集合，就是挑出tensor中的独立不重复元素。
                one_label_value = one_label.item()  # one_label 是一个tensor类型
                index =np.argwhere(labels_t2n == one_label_value)  # 先找出label为one_label的索引
                features_one_label = features[index,:]  # 找出所有标签为one_label 的features,此时的 features_one_label 是3维的，shape=[,1,]
                if index.size == 0:
                    print('index.size=0')
                    continue
                elif index.size ==1: # features_one_label 是一个三维的单个teosor,shape=torch.Size([1, 1, 512])
                    class_centroid_batch = torch.squeeze(features_one_label,dim=0) # 消除第一个维度后shape=torch.Size([1, 3])
                    class_centroid_batch = F.normalize(features_one_label,dim=1) # 按第二个维度标准化后还是一个二维的
                    class_centroid[one_label_value] = 0.5*class_centroid[one_label_value] + 0.5*class_centroid_batch # 一维加二维结果是二维
                    class_centroid[one_label_value] = F.normalize(torch.unsqueeze(class_centroid[one_label_value],dim=0), dim=1) # 这里取出的class_centroid[one_label_value]还是一个一维的,为其添加一个维度
                    #print('index.size=1')
                else: # features_one_label 是一个三维的teosor,shape=torch.Size([个数, 1, 512])
                    features_one_label_mean = torch.mean(features_one_label,dim=0) # 按第一个维度求均值后,shape = torch.Size([1, 512])
                    class_centroid_batch = F.normalize(features_one_label_mean,dim=1) # 按第二个维度标准化后还是一个二维的
                    class_centroid[one_label_value] = 0.5*class_centroid[one_label_value] + 0.5*class_centroid_batch # 一维加二维结果是二维
                    class_centroid[one_label_value] = F.normalize(torch.unsqueeze(class_centroid[one_label_value],dim=0), dim=1) # 这里取出的class_centroid[one_label_value]还是一个一维的,为其添加一个维度
    
    true_class_centroid = class_centroid
    
    return true_class_centroid
########################################################################################################
# # 遍历所有batch,使用移动指数平均获得每个类的centroid
# def get_class_centroid(net,train_loader,num_classes):
#     # 初始化类质心
#     class_centroid = torch.randn(num_classes,latent_variable_dim) # (0,1)正太分布
#     #class_centroid = F.normalize(class_centroid,dim=1).cuda() # 标准化初始类质心,并放到gpu上
#     for sample in train_loader:
#         data, labels = sample
#         data, labels = data.cuda(), labels.cuda()
           
#         # forward
#         features,logits= net(data) # out1和out2分别输出的feature和logits=-distance
#         with torch.no_grad():
#             labels_t2n = labels.cpu().numpy() # numpy.ndarray
#             # update clss centroids with a weighted moving average in each batch
#             for one_label in torch.unique(labels): # torch.unique()的功能类似于数学中的集合，就是挑出tensor中的独立不重复元素。
#                 one_label_value = one_label.item()  # one_label 是一个tensor类型
#                 index =np.argwhere(labels_t2n == one_label_value)  # 先找出label为one_label的索引
#                 features_one_label = features[index,:]  # 找出所有标签为one_label 的features,此时的 features_one_label 是3维的，shape=[,1,]
#                 if index.size == 0:
#                     continue
#                 elif index.size ==1: # features_one_label 是一个三维的单个teosor,shape=torch.Size([1, 1, 512])
#                     class_centroid_batch = torch.squeeze(features_one_label,dim=0) # 消除第一个维度后shape=torch.Size([1, 3])
#                     # class_centroid_batch = F.normalize(features_one_label,dim=1) # 按第二个维度标准化后还是一个二维的
#                     class_centroid[one_label_value] = 0.5*class_centroid[one_label_value] + 0.5*class_centroid_batch # 一维加二维结果是二维
#                     # class_centroid[one_label_value] = F.normalize(class_centroid[one_label_value], dim=1) # 按第二维度normalized
#                 else: # features_one_label 是一个三维的teosor,shape=torch.Size([个数, 1, 512])
#                     features_one_label_mean = torch.mean(features_one_label,dim=0) # 按第一个维度求均值后,shape = torch.Size([1, 512])
#                     # class_centroid_batch = F.normalize(features_one_label_mean,dim=1) # 按第二个维度标准化后还是一个二维的
#                     class_centroid[one_label_value] = 0.5*class_centroid[one_label_value] + 0.5*features_one_label_mean # 一维加二维结果是二维
#                     #class_centroid[one_label_value] = F.normalize(class_centroid[one_label_value], dim=1) # normalized
    
#     true_class_centroid =F.normalize(class_centroid,dim=1)  # 最后统一标准化
    
#     return true_class_centroid