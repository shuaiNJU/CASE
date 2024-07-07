from .resnet import get_resnet18,get_resnet34,get_resnet50,get_resnet101
# from .resnet import get_resnet_18

from .deconfnet import DeconfNet,get_h

# MDS
def get_feature_extractor(name:str, num_classes:int):
    if name == 'resnet18':
        net = get_resnet18(num_classes,include_top=False)
        #net.output_size = 1 * 512
    elif name == 'resnet34':
        net = get_resnet34(num_classes,include_top=False)
    elif name == 'resnet50':
        net = get_resnet50(num_classes,include_top=False)
    elif name == 'resnet101':
        net = get_resnet101(num_classes,include_top=False)
    else:
        raise RuntimeError('---> invalid network name: {}'.format(name))
    return net
##############################################################################
    
def get_deconf_net(name:str, num_classes:int):
    features_extractor =get_feature_extractor(name,num_classes)
    in_features = features_extractor.output_size  # 在resnet初始化的时候已经对output_size赋值了
    h = get_h(in_features,num_classes)  # h即EuclideanDeconf的一个实例化对象
    # print(in_features)
    return DeconfNet(features_extractor,in_features,h)
