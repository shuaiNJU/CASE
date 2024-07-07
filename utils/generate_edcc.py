import numpy as np

class_num = 10
latent_variable_dim = 512
G=1e-2  # 即lamda =0.01


def edcc_generation(u,v): # 传入的u是在每个batch中使用EMA计算出的tensor
        # 将tensor转变为numpy
        u = u.detach().cpu().numpy()  # detach().cpu()移至cpu 返回值是cpu上的Tensor
        # 生成pedcc
        num = u.shape[0] # 10或者100个样本数
        dd = np.zeros((num, num)) #(100,100):存储两个ui之间的距离
        for m in range(num):
            for n in range(num):
                # Compute the distance between ui
                if m <= n:
                    dd[m,n] = np.linalg.norm(u[m,:] - u[n,:])
                    dd[n,m] = dd[m,n]
        # 如果两个点之间的距离小于0.01，则将该距离设置为0.01，然后继续迭代
        dd[dd<1e-2] = 1e-2
        F = np.zeros((latent_variable_dim,num)) # F.shape=(512,100或10)
        for m in range(num):
            for n in range(num):
                # compute the resultant force vector 
                F[:,m] += (u[m,:]-u[n,:])/((dd[m][n])**3)
        F=F.T # 转置后F为[100,512]
        tmp_F=[]
        for i in range(F.shape[0]): #100
            tmp_F.append(np.dot(F[i],u[i])) # tmp_F是一个len(temp_F)=100的列表
        d = np.array(tmp_F).T.reshape(len(tmp_F), 1) # d是一个d.shape=(100,1)的数组numpy.ndarray
        # get the dot(u,f)u
        Fr = u*np.repeat(d, latent_variable_dim, 1) # 将重复操作施加到维度‘axis=1’上，相当于‘增加列数’,即d变为了[100,512]
        # Fr.shape=(100,512)
        # get the tangent vector
        Ft = F-Fr #
        u = u + v #(100,512)
        ll = np.sum(u**2,1)**0.5  # 计算每一个ui(每一行为一个ui)的模即|ui|。ll.shape=(100,)
        u = u/np.repeat(ll.reshape(ll.shape[0],1),latent_variable_dim,1) # normalized ui,归一化每个ui
        v = v + G*Ft
        
        return u,v