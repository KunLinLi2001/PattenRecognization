
#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from timeit import timeit
# 将列表中的数据切片读入矩阵
def Read(lines,m,n):
    A = np.zeros((m, n))
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        A[A_row:] = list[0:5]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    return A 
def DataInit(): 
    '''1.读取训练集和测试集'''
    f1,f2 = open('train.txt'),open('test.txt') # 打开训练集和测试集
    lines1,lines2 = f1.readlines(),f2.readlines() # 把全部数据文件读到一个列表lines中
    Line1,Line2 = len(lines1),len(lines2) # 读取训练集合测试集的行数
    A,B = Read(lines1,Line1,5),Read(lines2,Line2,5)
    return A,B

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
k = 10
train_sum = A.shape[0]
test_sum = B.shape[0]
train_id = A[:,0] # 训练集样本类别
test_id = B[:,0] # 测试集样本类别
A,B = np.delete(A,0,axis=1),np.delete(B,0,axis=1) # 删除标签
true,false = 0,0
# print(train_id)
for i in range(test_sum):
    # print(i)
    num = np.zeros(4) # 记录存放距离最近的k个点各自类别出现次数
    dis = np.zeros((train_sum,2))
    for j in range(train_sum):
        print(train_sum)
        print(j)
        dis[j,0] = train_id[j] # 训练集原属标签
        aaa = A[i,:]
        bbb = B[j,:]
        aa = np.power(aaa-bbb,2)
        dis[j,1] = np.sqrt(sum(aa)) # 欧式距离
    order = dis[np.lexsort(dis.T)] # 按距离从近到远升序排序
    # print(order)
    for top in range(k): # 找到距离该测试集前k近的训练集点
        index = order[top,0].astype(int)
        num[index] += 1
    Id = num.argmax()
    if Id == test_id[i]: # 如果预测标号等于我们的实际测试集标号
        true += 1
    else:
        false += 1
        print("错将测试集中的第%d组数据分成第%d类,正确类别为第%d类"%(i+1,Id,test_id[i]))
if(k==1):
    str = "最近邻法"
else:
    str = "k紧邻法"
print("基于%s对三类样本进行分类："%str)
print("正确个数：",true)
print("错误个数：",false)
print("准确率：",true/(true+false),'\n')

