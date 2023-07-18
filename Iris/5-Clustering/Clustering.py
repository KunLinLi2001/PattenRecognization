#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import math
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
    f = open("test.txt")
    lines = f.readlines() # 把全部数据文件读到一个列表lines中
    Line = len(lines) # 读取训练集合测试集的行数
    A = Read(lines,Line,5)
    return A


'''1.数据初始化'''
A = DataInit() # 四特征样本集(75*5矩阵)
id = np.mat(A[:,0]).transpose() # 记录样本集的标号(75*1列向量)

'''2.利用主成分分析方法进行样本降维'''

'''（1）计算协方差矩阵和均值向量'''
means = np.mean(A[:,1:],axis=0) # 特征均值
cov = np.cov(A[:,1:],rowvar=False) # 特征方差

'''（2）计算协方差矩阵的特征值、特征向量以及各主成分贡献率'''
val,vec = np.linalg.eig(cov) # 特征值与特征向量(原理：AX=λX )
val_rate = val/np.sum(val) # 贡献率

'''（3）可视化特征值的贡献率'''
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure()

x = ["特征1","特征2","特征3","特征4"]
y = val
colors = [ 'gold', 'lightskyblue','yellowgreen', 'lightcoral']
plt.pie(y,labels=x,colors=colors,autopct='%1.1f%%',shadow=False,startangle=90)
plt.axis('equal') # 显示为圆（避免比例压缩为椭圆）
plt.title('各特征成分所占比例')


'''（4）求解变换后的二维新特征'''
index = np.argsort(-val) # 降序排序返回下角标
index = index[:2] # 取出前两大特征
A_new = np.dot(A[:,1:],vec[index].transpose())
A_new = np.concatenate((id,A_new),axis=1) # 插入标签
# print(A_new) # 降维后的新数据集（75*3）

'''（5）降维后的新特征在新的特征空间画出样本点'''
plt.figure()
size = 10 # 点的大小
for item in A_new: 
    if item[0,0]==1.0: # 第一类
        plt.scatter(item[0,1], item[0,2],c='red',s=size)
    elif item[0,0]==2.0: # 第二类
        plt.scatter(item[0,1], item[0,2],c='blue',s=size)
    elif item[0,0]==3.0: # 第三类
        plt.scatter(item[0,1], item[0,2],c='yellow',s=size)
plt.title("降维后的新特征在样本空间的分布")
# plt.show()

'''3.k均值聚类实现'''
'''（1）算法初始化'''
A = A_new
# 凭经验选择的代表点(选择上述散点图近似质心的点)
z1 = [A[18,1],A[18,2]]
z2 = [A[49,1],A[49,2]]
z3 = [A[55,1],A[55,2]]

# 用前k个样本点作为代表点
# np.random.shuffle(A) # 随机打乱训练集
# z1 = [A[0,1],A[0,2]]
# z2 = [A[1,1],A[1,2]]
# z3 = [A[2,1],A[2,2]]

# 随机选取代表点
# A_random = A
# np.random.shuffle(A_random) # 随机打乱训练集
# A1,A2,A3 = A_random[0:25],A_random[25:50],A_random[50:75] # 分成三类
# means = np.mean(A[:,1:],axis=0) # 特征均值
# zz1,zz2,zz3 = np.mean(A1,axis=0),np.mean(A2,axis=0),np.mean(A3,axis=0)
# z1 = [zz1[0,1],zz1[0,2]]
# z2 = [zz2[0,1],zz2[0,2]]
# z3 = [zz3[0,1],zz3[0,2]]


def get_dis(item,z):
    return math.sqrt(math.pow(item[0,1]-z[0],2)+math.pow(item[0,2]-z[1],2))
def get_center(res):
    x,y = 0,0
    num = len(res)
    for item in res:
        x += item[0,1]
        y += item[0,2]
    x,y = x/num, y/num
    return [x,y]
count =1
'''（2）根据聚类中心进行聚类的迭代过程'''
while 1:
    res1,res2,res3 = [],[],[]
    for item in A:
        dis = [get_dis(item,z1),get_dis(item,z2),get_dis(item,z3)]
        min_index = dis.index(min(dis))+1
        if min_index == 1:
            res1.append(item)
        elif min_index == 2:
            res2.append(item)
        elif min_index == 3:
            res3.append(item)
    '''计算新的聚类中心'''
    old1 ,old2 ,old3 =z1,z2,z3
    z1,z2,z3 =get_center(res1),get_center(res2),get_center(res3)
    if old1 == z1 and old2 == z2 and old3 == z3:
        print("质心在第%d次迭代中确定下来，迭代完毕！"%count)
        break
    else:
        print("第%d次迭代中质心发生变化，继续迭代"%count)
        count += 1
    print("now:",z1,z2,z3)
    print("old:",old1,old2,old3)
print("最终三类质心为：")
print(z1,z2,z3)


'''（3）测试准确率'''
def count_accuracy(true,false,res,id):
    for item in res:
        if item[0,0] == id:
            true += 1
        else:
            false += 1
    return true,false
true,false = 0,0
true,false = count_accuracy(true,false,res1,id=1)
true,false = count_accuracy(true,false,res2,id=2)
true,false = count_accuracy(true,false,res3,id=3)
print("基于k均值算法对三类样本进行分类：")
print("正确个数：",true)
print("错误个数：",false)
print("准确率：",true/(true+false),'\n')

'''（4）画出分类后的样本点'''
plt.figure()
for item in res1:
    plt.scatter(item[0,1], item[0,2],c='red',s=size) # 第一类
for item in res2:
    plt.scatter(item[0,1], item[0,2],c='blue',s=size) # 第二类
for item in res3:
    plt.scatter(item[0,1], item[0,2],c='yellow',s=size) # 第三类
plt.title("分类后的样本点在样本空间的分布")
plt.show()