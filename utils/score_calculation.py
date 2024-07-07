from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import losses

to_np = lambda x: x.data.cpu().numpy() # 转换为numpy.ndarray类型
concat = lambda x: np.concatenate(x, axis=0) # 合并成一行

###############
def get_calibration_scores(net, loader):
    logits_list = []
    labels_list = []

    # from common.loss_function import _ECELoss
    ece_criterion = losses.ECELoss(n_bins=15)
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]

            data = data.cuda()
            label = target.cuda()
            # logits = net(data)
            features,logits,quotients,_ = net(data)

            logits_list.append(quotients)
            labels_list.append(label)
        quotients = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    # ece_error = ece_criterion(logits, labels, args.T)
    ece_error = ece_criterion(quotients, labels)
    return ece_error
#########################################################################################
# def get_msp_score(loader,net, in_dist=False):  # 在得分函数内部选择使用MSP/Energy/MSP+OE
#     _score = []
#     _right_score = []
#     _wrong_score = []

#     with torch.no_grad():
#         for batch in loader:

#             if loader.dataset.labeled:
#                 data,target= batch
#             else:
#                 data = batch
#             data = data.cuda()

#             output = net(data)  #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
#             smax = to_np(F.softmax(output, dim=1))  # MSP  [batch,10]
#             # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
#             _score.append(np.max(smax, axis=1))  #np.max()直接取出每行(axis=1)中最大的值

#             if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
#                 preds = np.argmax(smax, axis=1)  # axis=0 代表列 , axis=1 代表行
#                 targets = target.numpy().squeeze()
#                 right_indices = preds == targets 
#                 wrong_indices = np.invert(right_indices)
                
#                 _right_score.append(np.max(smax[right_indices], axis=1))
#                 _wrong_score.append(np.max(smax[wrong_indices], axis=1))

#     if in_dist:
#         return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return concat(_score).copy()
######################################################################################
def get_msp_score(loader,deconf_net, in_dist=False):  # 在得分函数内部选择使用MSP/Energy/MSP+OE
    deconf_net.eval()
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch in loader:

            if loader.dataset.labeled:
                data,target= batch
            else:
                data = batch
            data = data.cuda()

            features,logits,quotients,_ = deconf_net(data) # 三个输出：features, logits=-distance，h/g
            # output = net(data)  #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
            smax = to_np(F.softmax(logits, dim=1))  # MSP  [batch,10]
            # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
            _score.append(np.max(smax, axis=1))  #np.max()直接取出每行(axis=1)中最大的值

            if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
                preds = np.argmax(smax, axis=1)  # axis=0 代表列 , axis=1 代表行
                targets = target.numpy().squeeze()
                right_indices = preds == targets 
                wrong_indices = np.invert(right_indices)
                
                _right_score.append(np.max(smax[right_indices], axis=1))
                _wrong_score.append(np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()
#########################################################################################
# def get_energy_score(loader,net,T,in_dist=False):
#     _score = []
#     _right_score = []
#     _wrong_score = []

#     with torch.no_grad():
#         for batch in loader:
            
#             if loader.dataset.labeled:
#                 data,target= batch
#             else:
#                 data = batch
#             data = data.cuda()

#             _,output = net(data)  #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
#             smax = to_np(F.softmax(output, dim=1))  # MSP  [batch,10]

#             #temperature=1.0 , energy score本身就是一个负数
#             _score.append(-to_np(-(T*torch.logsumexp(output / T, dim=1)))) # negative energy score 是一个正数
#             # print(_score) 输出的每一个分数都是正分数
            
#             if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
#                 preds = np.argmax(smax, axis=1)  # axis=0 代表列 , axis=1 代表行
#                 targets = target.numpy().squeeze()
#                 right_indices = preds == targets 
#                 wrong_indices = np.invert(right_indices)
                
#                 _right_score.append(np.max(smax[right_indices], axis=1))
#                 _wrong_score.append(np.max(smax[wrong_indices], axis=1))

#     if in_dist:
#         return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return concat(_score).copy()
########################################################################################
def get_energy_score(loader,deconf_net,T,in_dist=False):
    deconf_net.eval()
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch in loader:
            
            if loader.dataset.labeled:
                data,target= batch
            else:
                data = batch
            data = data.cuda()

            features,logits,quotients,_ = deconf_net(data) # 三个输出：features, logits=-distance，h/g
            # _,output = net(data)  #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
            smax = to_np(F.softmax(logits, dim=1))  # MSP  [batch,10]

            #temperature=1.0 , energy score本身就是一个负数
            _score.append(-to_np(-(T*torch.logsumexp(logits / T, dim=1)))) # negative energy score 是一个正数
            # print(_score) 输出的每一个分数都是正分数
            
            if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
                preds = np.argmax(smax, axis=1)  # axis=0 代表列 , axis=1 代表行
                targets = target.numpy().squeeze()
                right_indices = preds == targets 
                wrong_indices = np.invert(right_indices)
                
                _right_score.append(np.max(smax[right_indices], axis=1))
                _wrong_score.append(np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()
#####################################################
# 没有使用input preprocessing
# def get_improved_energy_score(loader,net,temp,in_dist=False):
#     _score = []
#     _right_score = []
#     _wrong_score = []

#     with torch.no_grad():
#         for batch in loader:
            
#             if loader.dataset.labeled:
#                 data,target= batch
#             else:
#                 data = batch
#             data = data.cuda()

#             _,output = net(data)  #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
#             smax = to_np(F.softmax(output, dim=1))  # MSP  [batch,10]

#             #temperature=1.0 , energy score本身就是一个负数
#             _score.append(-to_np(-(torch.logsumexp(temp*output , dim=1)))) # negative energy score 是一个正数
#             # print(_score) 输出的每一个分数都是正分数
            
#             if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
#                 preds = np.argmax(smax, axis=1)  # axis=0 代表列 , axis=1 代表行
#                 targets = target.numpy().squeeze()
#                 right_indices = preds == targets 
#                 wrong_indices = np.invert(right_indices)
                
#                 _right_score.append(np.max(smax[right_indices], axis=1))
#                 _wrong_score.append(np.max(smax[wrong_indices], axis=1))

#     if in_dist:
#         return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return concat(_score).copy()
#################################################################################
#使用了input prepeocessing: 像odin 文中那样
# def get_improved_energy_score(loader,deconf_net,noiseMagnitude,in_dist=False):
#     _score = []
#     _right_score = []
#     _wrong_score = []

#     # with torch.no_grad():
#     for batch in loader:
        
#         if loader.dataset.labeled:
#             data,target= batch
#         else:
#             data = batch
#         data = data.cuda()

#         inputs = Variable(data,requires_grad = True)  # 把tensor放到篮子Variable中, requires_grad是参不参与误差反向传播, 要不要计算梯度
#         # forward
#         features,logits,quotients = deconf_net(inputs) # 三个输出：features, logits=-distance，h/g
#         smax = to_np(F.softmax(logits, dim=1))  # MSP  [batch,10]

#         # backward
#         criterion = nn.CrossEntropyLoss()
#         maxIndexTemp = np.argmax(logits.data.cpu().numpy(), axis=1)

#         # # Using temperature scaling
#         # outputs =outputs 

#         labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
#         loss = criterion(logits, labels)
#         loss.backward()

#         # Normalizing the gradient to binary in {0, 1}
#         gradient =  torch.ge(inputs.grad.data, 0)
#         gradient = (gradient.float() - 0.5) * 2
        
#         #根据ID是cifar10还是cifar100使用不同的标准差
#         # cifar10的标准差
#         # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
#         # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
#         # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
#         # cifar100的标准差
#         gradient[:,0] = (gradient[:,0] )/(0.2673)
#         gradient[:,1] = (gradient[:,1] )/(0.2564)
#         gradient[:,2] = (gradient[:,2] )/(0.2762)
        
#         # Adding small perturbations to images
#         tempInputs = torch.add(input=inputs.data, alpha=-noiseMagnitude, other=gradient)
#         # forward
#         features,logits,quotients = deconf_net(Variable(tempInputs))

#         # Calculating the confidence after adding perturbations
#         smin= logits.max(dim=1)[0]  #  this is the minimum distance score：因为输出的距离是一个负距离，所以最大负距离=最小正距离
#         ##########################################################
#         smin = to_np(smin) # 将tensor类型转换为numpy.ndarray
#         _score.append(smin)  # 将这一个batch的score存储到列表_score中
#         #_score.append(-to_np(-(torch.logsumexp(temp*outputs , dim=1)))) # negative energy score 是一个正数
#         # print(_score) 输出的每一个分数都是正分数
        
#         if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
#             preds = np.argmax(smax, axis=1)  # axis=0 代表列 , axis=1 代表行
#             targets = target.numpy().squeeze()
#             right_indices = preds == targets 
#             wrong_indices = np.invert(right_indices)
            
#             _right_score.append(np.max(smax[right_indices], axis=1))
#             _wrong_score.append(np.max(smax[wrong_indices], axis=1))

#     if in_dist:
#         return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return concat(_score).copy()
###############################################
# 使用godin那样的输入预处理
def get_energy_score_godin_pro(loader,deconf_net,T,noiseMagnitude,in_dist=False):
    deconf_net.eval()
    _score = []
    _right_score = []
    _wrong_score = []

    # with torch.no_grad():
    for batch in loader:
        
        if loader.dataset.labeled:
            data,target= batch
        else:
            data = batch
        data = data.cuda()

        
        # forward
        data.requires_grad = True
        features,logits1,quotients = deconf_net(data) # 三个输出：features, logits=-distance，h/g
       
       # backward
        max_scores = T*torch.logsumexp(logits1/ T, dim=1)
        # max_scores, _ = torch.max(logits1, dim=1)
        max_scores.backward(torch.ones(len(max_scores)).cuda())

        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2
       
        #根据ID是cifar10还是cifar100使用不同的标准差
        # cifar10的标准差
        # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
        # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
        # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
        # cifar100的标准差
        gradient[:,0] = (gradient[:,0] )/(0.2673)
        gradient[:,1] = (gradient[:,1] )/(0.2564)
        gradient[:,2] = (gradient[:,2] )/(0.2762)
        
        # input preprocessing and forward
        tmpInputs = torch.add(input=data.detach(),alpha= noiseMagnitude, other=gradient)
        with torch.no_grad():
            features,logits,quotients = deconf_net(tmpInputs)

        # max_sco = to_np(torch.max(logits, dim=1)[0])
        max_sco = to_np(T*torch.logsumexp(logits / T, dim=1))
        _score.append(max_sco) 
        
        if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
            #quotients = quotients.cpu().detach().numpy()
            logits1 = logits1.cpu().detach().numpy()
            preds = np.argmax(logits1, axis=1)  # axis=0 代表列 , axis=1 代表行
            targets = target.numpy().squeeze()
            right_indices = preds == targets 
            wrong_indices = np.invert(right_indices)
            
            _right_score.append(np.max(logits1[right_indices], axis=1))
            _wrong_score.append(np.max(logits1[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()
############################################################################################
# 使用godin那样的输入预处理
def get_like_godin_score(loader,deconf_net,noiseMagnitude,in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    # with torch.no_grad():
    for batch in loader:
        
        if loader.dataset.labeled:
            data,target= batch
        else:
            data = batch
        data = data.cuda()

        
        # forward
        data.requires_grad = True
        features,logits1,quotients = deconf_net(data) # 三个输出：features, logits=-distance，h/g
       
       # backward
        max_scores, _ = torch.max(logits1, dim=1)
        max_scores.backward(torch.ones(len(max_scores)).cuda())

        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2
       
        #根据ID是cifar10还是cifar100使用不同的标准差
        # cifar10的标准差
        # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
        # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
        # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
        # cifar100的标准差
        gradient[:,0] = (gradient[:,0] )/(0.2673)
        gradient[:,1] = (gradient[:,1] )/(0.2564)
        gradient[:,2] = (gradient[:,2] )/(0.2762)
        
        # input preprocessing and forward
        tmpInputs = torch.add(input=data.detach(),alpha= noiseMagnitude, other=gradient)
        features,logits,quotients = deconf_net(tmpInputs)

        max_sco = to_np(torch.max(logits, dim=1)[0])
        _score.append(max_sco) 
        
        if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
            #quotients = quotients.cpu().detach().numpy()
            logits1 = logits1.cpu().detach().numpy()
            preds = np.argmax(logits1, axis=1)  # axis=0 代表列 , axis=1 代表行
            targets = target.numpy().squeeze()
            right_indices = preds == targets 
            wrong_indices = np.invert(right_indices)
            
            _right_score.append(np.max(logits1[right_indices], axis=1))
            _wrong_score.append(np.max(logits1[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()
################################################################################################
def get_MDS_score2(loader,net,dis_scale,in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch in loader:
            
            if loader.dataset.labeled:
                data,target= batch
            else:
                data = batch
            data = data.cuda()

            _,output = net(data)  #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
            smax = to_np(F.softmax(output, dim=1))  # MSP  [batch,10]

            #temperature=1.0 , energy score本身就是一个负数
            _score.append(-to_np(-(torch.logsumexp(dis_scale*output , dim=1)))) # negative energy score 是一个正数
            # print(_score) 输出的每一个分数都是正分数
            
            if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
                preds = np.argmax(smax, axis=1)  # axis=0 代表列 , axis=1 代表行
                targets = target.numpy().squeeze()
                right_indices = preds == targets 
                wrong_indices = np.invert(right_indices)
                
                _right_score.append(np.max(smax[right_indices], axis=1))
                _wrong_score.append(np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()

#####################################################
def get_MDS_score(loader,net, in_dist=False):
    net.eval()
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch in loader:
            
            if loader.dataset.labeled:
                data,target= batch
            else:
                data = batch
            data = data.cuda()
            ##########################################################
            #features,logit = net(data)  # 输出的 logit = -distance,其shape=torch.Size([128,10或100]),每一行代表一个样本：到每一个pedcc的距离值
            features,logit,quotients  = net(data)
            smin= logit.max(dim=1)[0]  #  this is the minimum distance score：因为输出的距离是一个负距离，所以最大负距离=最小正距离
            ##########################################################
            smin = to_np(smin) # 将tensor类型转换为numpy.ndarray
            _score.append(smin)  # 将这一个batch的score存储到列表_score中
            
   
            if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
                preds= logit.max(dim=1)[1]
                preds = to_np(preds)
                targets = target.numpy().squeeze() # squeeze() 从数组的形状中删除单维度条目,即把shape中为1的维度去掉
                right_indices = preds == targets  # right_indices= array([False,True,...])这种形式
                wrong_indices = np.invert(right_indices)

                _right_score.append(smin[right_indices]) # 存储预测正确的样本所对应的score
                _wrong_score.append(smin[wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()

####################################################################
# def get_MDS_and_mean_MDS_score(loader,net, dis_scale,num_classes,in_dist=False):
#     _score = []
#     _right_score = []
#     _wrong_score = []

#     with torch.no_grad():
#         for batch in loader:
            
#             if loader.dataset.labeled:
#                 data,target= batch
#             else:
#                 data = batch
#             data = data.cuda()
#             ##########################################################
#             features,logit = net(data)  # 输出的 logit = -distance,其shape=torch.Size([128,10或100]),每一行代表一个样本：到每一个pedcc的距离值
#             logit = dis_scale * logit  # -ES|fx-uj|
#             smin= logit.max(dim=1)[0]  #  this is the minimum distance score：因为输出的距离是一个负距离，所以最大负距离=最小正距离
#             smin_to_np =to_np(smin)
            
#             sim_exp = torch.exp(smin)  #e^-ES|fx-ui|
#             log_logit = (torch.logsumexp(logit , dim=1))
#             # logit_exp =torch.exp(logit)
#             # logit_exp_sum_mean = torch.sum(logit_exp,dim=1)/num_classes
#             mds_score = sim_exp + log_logit
#             ##########################################################
#             score = to_np(mds_score) # 将tensor类型转换为numpy.ndarray
#             _score.append(score)  # 将这一个batch的score存储到列表_score中
            
   
#             if in_dist: # 如果是D_test_in，还需要计算出_right_score，_wrong_score
#                 preds= logit.max(dim=1)[1]
#                 preds = to_np(preds)
#                 targets = target.numpy().squeeze() # squeeze() 从数组的形状中删除单维度条目,即把shape中为1的维度去掉
#                 right_indices = preds == targets  # right_indices= array([False,True,...])这种形式
#                 wrong_indices = np.invert(right_indices)

#                 _right_score.append(smin_to_np[right_indices]) # 存储预测正确的样本所对应的score
#                 _wrong_score.append(smin_to_np[wrong_indices])

#     if in_dist:
#         return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return concat(_score).copy()
####################################################################
# def get_odin_score(loader, net, T, noise, in_dist=False):
#     _score = []
#     _right_score = []
#     _wrong_score = []

#     net.eval()
#     for sample in loader:
        
#         if loader.dataset.labeled:
#             data,target = sample
#         else:
#             data = sample
#         data = data.cuda()
#         data = Variable(data, requires_grad = True)  # 把tensor放到篮子Variable中, requires_grad是参不参与误差反向传播, 要不要计算梯度

#         output = net(data)
#         smax = to_np(F.softmax(output, dim=1))

#         odin_score = ODIN(data,output,net,T,noise) #将原始data作为输入,output为logit
#         _score.append(np.max(odin_score, 1))

#         if in_dist:
#             preds = np.argmax(smax, axis=1)
#             targets = target.numpy().squeeze()
#             right_indices = preds == targets
#             wrong_indices = np.invert(right_indices)

#             _right_score.append(np.max(smax[right_indices], axis=1))
#             _wrong_score.append(np.max(smax[wrong_indices], axis=1))

#     if in_dist:
#         return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return concat(_score).copy()
###############################################################################
def get_odin_score(loader,deconf_net, T, noise, in_dist=False):
    deconf_net.eval()
    _score = []
    _right_score = []
    _wrong_score = []

    deconf_net.eval()
    for sample in loader:
        
        if loader.dataset.labeled:
            data,target = sample
        else:
            data = sample
        data = data.cuda()
        data = Variable(data, requires_grad = True)  # 把tensor放到篮子Variable中, requires_grad是参不参与误差反向传播, 要不要计算梯度

        features,logits,quotients,_ = deconf_net(data) # 三个输出：features, logits=-distance，h/g
        # output = net(data)
        smax = to_np(F.softmax(logits, dim=1))

        odin_score = ODIN(data,logits,deconf_net,T,noise) #将原始data作为输入,output为logit
        _score.append(np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(np.max(smax[right_indices], axis=1))
            _wrong_score.append(np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()

def ODIN(inputs, outputs, deconf_net, temper, noiseMagnitude):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels) # 集logS(x;T)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0) # 逐元素比较torch.ge(a,b)比较a,b的大小,a为张量，b可以为和a相同形状的张量，也可以为一个常数。>=0赋值为1
    gradient = (gradient.float() - 0.5) * 2
    # cifar10
    # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    # CIFAR100
    gradient[:,0] = (gradient[:,0] )/(0.2673)
    gradient[:,1] = (gradient[:,1] )/(0.2564)
    gradient[:,2] = (gradient[:,2] )/(0.2762)
   
    # Adding small perturbations to images
    tempInputs = torch.add(input=inputs.data, alpha=-noiseMagnitude, other=gradient)
    features,logits,quotients,_ = deconf_net(Variable(tempInputs)) # 四个输出：features, logits=-distance，h/g
    #outputs = model(Variable(tempInputs))
    logits = logits / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = logits.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    for batch_id,sample in enumerate(test_loader):
        if test_loader.dataset.labeled:
            data,target = sample  # 对data在下面进行统一处理
            
            target = target.cuda()
            target = Variable(target)
        else:
            data = sample
        
        data = data.cuda()
        data = Variable(data, requires_grad = True)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag() # torch.mm(a, b) 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0) #torch.ge(a,b)比较a，b的大小，a为张量，b可以为和a相同形状的张量，也可以为一个常数
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        
        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        
        if batch_id ==0:
            print(noise_gaussian_score)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())  # 去掉了分数的负号,原始论文就没带负号。
        
    return np.asarray(Mahalanobis, dtype=np.float32)

