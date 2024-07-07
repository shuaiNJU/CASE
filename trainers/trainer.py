import torch
import torch.nn.functional as F
import losses

device = 1
torch.cuda.set_device(device)
# classify
class ClassifierTrainer():
    
    def __init__(
        self,
        deconf_net,
        train_loader,
        optimizer,
        h_optimizer,
        scheduler,
        h_scheduler,
        compact_loss,
        disper_loss
    ):
        self.deconf_net = deconf_net
        self.train_loader = train_loader
        
        self.optimizer = optimizer
        self.h_optimizer = h_optimizer
        self.scheduler = scheduler
        self.h_scheduler = h_scheduler
        ###################################
        self.compact_loss = compact_loss # 创建类Compatness_loss的一个实例化对象,括号中参数的个数与类中__init__()中参数个数一样
            # 因为原类init中有一个默认的初始化参数entropic_scale=1.0，这里无需在对其进行赋值了
        self.disper_loss = disper_loss
        ###################################
        
    def train_epoch(self):
        self.deconf_net.train()   # enter train mode
        total, correct = 0, 0 # 记录总数和预测对的个数
        total_loss = 0.0
        
        for sample in self.train_loader:
            data, target = sample
            data, target = data.cuda(), target.cuda()
            #print('target=',target.data)
            # forward
            features,logits,quotients,class_centroid = self.deconf_net(data) # 三个输出：features, logits=-distance，h/g
            #######################################
             # Intra_class_compactness_loss
            c_loss= self.compact_loss(quotients,target)  # 等价于交叉熵损失
            d_loss= self.disper_loss(class_centroid)  # 使距离差变大
            loss = c_loss+d_loss
            # loss = d_loss
            #######################################
           
            self.optimizer.zero_grad() # 1、梯度清零
            self.h_optimizer.zero_grad()
            loss.backward()  # 2、backward:反向传播计算得到每个参数的梯度值
            self.optimizer.step() #3、最后通过梯度下降执行一步参数更新
            self.h_optimizer.step()
            #self.scheduler.step()
            
            _, pred = quotients.max(dim=1) # quotients中最大的就是2-norm最小的
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0) # target.size(0) = train set中的图片总个数
                correct += pred.eq(target).sum().item()  #预测对的个数
        
        self.h_scheduler.step()
        self.scheduler.step()
        # average on batch: len(train_loader)=batch的总个数。 average on sample:len(train_loader.dataset)= 样本总个数
        print('-train:[loss: {:.4f} | accuracy: {:.4f}%]'.format(total_loss / len(self.train_loader), 100. * correct / total))
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_accuracy': correct / total
        }
        
        return metrics