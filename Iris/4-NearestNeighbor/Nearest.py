#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from timeit import timeit
# from sklearn.decomposition import PCA
# import sklearn
# 将列表中的数据切片读入矩阵
def Read(lines,m,n):
    A = np.zeros((m, n))
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        A[A_row:] = list[0:5]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    return A 
# 数据集初始化
def DataInit(): 
    '''1.读取训练集和测试集'''
    f1,f2 = open('train.txt'),open('test.txt') # 打开训练集和测试集
    lines1,lines2 = f1.readlines(),f2.readlines() # 把全部数据文件读到一个列表lines中
    Line1,Line2 = len(lines1),len(lines2) # 读取训练集合测试集的行数
    A,B = Read(lines1,Line1,5),Read(lines2,Line2,5)
    return A,B
# k近邻法函数
def k_nearest(A,B,show,k):
    train_sum = A.shape[0]
    test_sum = B.shape[0]
    train_id = A[:,0] # 训练集样本类别
    test_id = B[:,0] # 测试集样本类别
    A,B = np.delete(A,[0],axis=1),np.delete(B,[0],axis=1) # 删除标签
    true,false = 0,0
    for i in range(test_sum):
        num = np.zeros(4) # 记录存放距离最近的k个点各自类别出现次数
        dis = np.zeros((train_sum,2))
        for j in range(train_sum):
            dis[j,0] = train_id[j] # 训练集原属标签
            dis[j,1] = np.sqrt(sum(np.power(A[j,:]-B[i,:],2))) # type: ignore # 欧式距离
        order = dis[np.lexsort(dis.T)] # 按距离从近到远升序排序
        for top in range(k): # 找到距离该测试集前k近的训练集点
            index = order[top,0].astype(int)
            num[index] += 1
        Id = num.argmax()
        if Id == test_id[i]: # 如果预测标号等于我们的实际测试集标号
            true += 1
        else:
            false += 1
            if show == True:
                print("错将测试集中的第%d组数据分成第%d类,正确类别为第%d类"%(i+1,Id,test_id[i]))
    if k==1:
        str = "最近邻法"
    else:
        str = "k近邻法"
    if show == True:
        print("基于%s对三类样本进行分类："%str)
        print("正确个数：",true)
        print("错误个数：",false)
        print("准确率：",true/(true+false),'\n')
    else:
        return true/(true+false)
# 对训练集进行剪切
def cut_train(A,B,ok):
    # ok = True
    k = 1
    train_sum = A.shape[0]
    test_sum = B.shape[0]
    train_id = A[:,0] # 训练集样本类别
    test_id = B[:,0] # 测试集样本类别
    wrong = [] # 记录错误样本的标号
    for i in range(test_sum):
        num = np.zeros(4) # 记录存放距离最近的k个点各自类别出现次数
        dis = np.zeros((train_sum,2))
        for j in range(train_sum):
            dis[j,0] = train_id[j] # 训练集原属标签
            dis[j,1] = np.sqrt(sum(np.power(A[j,1:]-B[i,1:],2))) # type: ignore # 欧式距离
        order = dis[np.lexsort(dis.T)] # 按距离从近到远升序排序
        for top in range(k): # 找到距离该测试集前k近的训练集点
            index = order[top,0].astype(int)
            num[index] += 1
        Id = num.argmax()
        if Id != test_id[i]: # 如果预测标号不等于我们的实际测试集标号
            wrong.append(i)
            ok = False
    B = np.delete(B,wrong,axis=0)
    return A,B,ok
# 训练校验过程
def get_train_cut(A_random):
    # 将打乱的训练集拆分成五组进行迭代训练
    ok = False # 全部剪辑完毕则为yes
    count = 0
    '''在这里进行一个创新改进，连续十轮随机剪辑后没有错分对象才可以认为剪辑完毕'''
    while(not ok):
        ok = True
        np.random.shuffle(A_random) # 随机打乱训练集
        train_sum = A_random.shape[0]
        divide = train_sum // 5
        A1,A2,A3,A4,A5 = A_random[0:divide,:],A_random[divide:2*divide,:],A_random[2*divide:3*divide,:],A_random[3*divide:4*divide,:],A_random[4*divide:train_sum ,:]
        # 修正的过程
        A2,A1,ok = cut_train(A2,A1,ok)
        A3,A2,ok = cut_train(A3,A2,ok)
        A4,A3,ok = cut_train(A4,A3,ok)
        A5,A4,ok = cut_train(A5,A4,ok)
        A1,A5,ok = cut_train(A1,A5,ok)
        A_random = np.concatenate((A1,A2,A3,A4,A5),axis=0)
        if ok==True:
            count +=1
            if count == 10:
                ok = True
            else:
                ok = False
        else:
            count = 0
    return A_random
# 对剪切过的训练集进行压缩
def get_train_compress(A_cut):
    length = A_cut.shape[0]
    Store = np.array([A_cut[0,:]]) # 剪辑后的训练集第一组数据默认放入Store
    Grabbag = A_cut[1:length,:] # 剪辑后的训练集除第一组以外均放入Grabbag
    count = 0 # 计数连续多少个样本测试正确
    Len = Grabbag.shape[0] # 当前Grabbag有多少数据
    while 1: 
        test = np.array([Grabbag[0,:]]) # 取出待测试样本
        Grabbag = np.delete(Grabbag,0,axis=0) # 从Grabbag中去除
        cur = k_nearest(Store,test,False,1)
        if cur == 1: # 分类正确
            Grabbag = np.insert(Grabbag,Grabbag.shape[0],test,axis=0) # 尾插回去
            count += 1
        else : # 分类错误
            Store = np.insert(Store,Store.shape[0],test,axis=0) # 转入Store
            count = 0
            Len = Grabbag.shape[0] # 更新

        if Grabbag.shape[0] == 0 or count == Len:
            break
    return Store
'''
    一、数据集初始化：
    其中A,B分别为训练集和测试集
    并将训练集扩充成5倍，增大训练集数目
'''
A,B = DataInit()
# t = timeit('DataInit(k)', 'from __main__ import DataInit', number=1000)
# print(t)
A = np.concatenate((A,A,A,A,A),axis=0) # 将训练集扩充成5倍
B = np.concatenate((B,B,B),axis=0) # 测试集扩充成3倍
'''
    二、最近邻法：
    计算测试集点到训练集每一个点的距离，
    距离测试点最近的训练集点的类别作为该测试集的预估值。
'''
k_nearest(A,B,True,1) # k=1的k近邻就是最近邻法
'''
    三、k近邻法：
    计算测试集点到训练集每一个点的距离，
    统计距离测试点最近的k个训练集点,
    这k个训练集点出现的类别次数最多的那类作为该测试集的预估值。 
'''
k_nearest(A,B,True,30)
'''
    四、通过图表可视化观察k从1到200变化时
    对应的准确率的变化
'''
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
x = list(range(1,201))
y = []
for i in range(1,201):
    point = k_nearest(A,B,False,i)
    y.append(point)
plt.plot(x,y,label="测试准确率")
plt.xlabel("近邻法的k值")
plt.ylabel("测试准确率")
plt.title("k从1到200变化时近邻法识别准确率的变化")
plt.show()
'''
    五、剪辑近邻法
    先对训练集进行多重剪辑处理，得到一个新的训练集
'''
A_cut = get_train_cut(A) # 剪辑训练集
cut = A.shape[0]-A_cut.shape[0]
print("训练集本有%d组数据,剪辑掉%d组,最终训练集剩余%d组"%(A.shape[0],cut,A_cut.shape[0]))
k_nearest(A_cut,B,True,1) # 使用最近邻法进行分类
'''
    六、压缩近邻法
    在Grabbag中取出第i个样本用Store中的当前样本集按最近邻法分类。
    若分类错误，则将该样本从Grabbag转入Store中，
    若分类正确，则将该样本放回Grabbag中，对Grabbag中所有样本重复上述过程。
'''
A_compress = get_train_compress(A_cut)
compress = A_cut.shape[0]-A_compress.shape[0]
print("剪辑后的训练集本有%d组数据,压缩掉%d组,最终训练集剩余%d组"%(A_cut.shape[0],compress,A_compress.shape[0]))
k_nearest(A_compress,B,True,1)