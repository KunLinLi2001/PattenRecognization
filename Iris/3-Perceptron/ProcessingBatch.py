#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# 将列表中的数据切片读入矩阵
def Read(lines,m,n):
    A = np.zeros((m, n))
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        A[A_row:] = list[0:5]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    return A 
# 计算准确率
def count_accuracy(true,false,id,Id):
    if id==Id:
        true+=1
    else:
        false+=1
    return true,false
# 利用批处理的梯度下降算法设计的感知器
def processing_batch(A1,A2,ID,del_line,w,p):
    '''
    (1)去除无用的数据
    例如目前需要绘制第一类和第二类的分类面，选取前三个特征进行分类
    就需要去除第一行（标签）和第五行的数据（第四个特征）
    '''
    A1 = np.delete(A1,[0,del_line],axis=1)
    A2 = np.delete(A2,[0,del_line],axis=1)
    '''
    (2)样本增广化、规范化处理
    '''
    # 样本增广化
    One = np.ones(25) # 1*25的行向量值均为1
    A1 = np.insert(A1,0,values=One,axis=1) # 在第0列插入One向量（变成了列向量）
    A2 = np.insert(A2,0,values=One,axis=1)
    # 样本规范化
    A = np.concatenate((A1,-1*A2),axis=0) # A1与A2矩阵拼接在一起
    # print(A,'\n')
    '''
    (3)采用批处理的梯度下降算法进行迭代运算
    计算权向量w和迭代次数k
    w为4*1的向量,初始化时设定为[1,1,1,1]^T
    any(data<=0 for data in y) 意思为在y矩阵里是否存在元素<=0
    设定学习率p
    迭代的同时绘制出迭代次数与惩罚值的散点图来描绘其收敛过程
    '''
    k = 1 # 迭代次数
    # 开始迭代！！
    print("采用批处理的梯度下降算法迭代计算第%d类和%d类的权向量:"%(ID[0],ID[1]))
    while 1:
        J = 0; 
        y = np.dot(A,w) # 计算样本训练值(错误样本训练值为负数)
        '''找到结果为负数的结果，求和后再取相反数作为惩罚J'''
        for i in range(0,50):
            item = y[i]
            if item <= 0:
                J = J - item
                w = w + p * np.mat(A[i,:]).transpose() 
        print("第%d次迭代中,求解惩罚值为%d"%(k,J))
        print("更新权向量值为:",w.transpose())
        # print(w.transpose)
        '''判断是否迭代完毕'''
        if all(data>0 for data in y):
            print("迭代完毕！")
            break
        '''下一轮的迭代'''
        k = k + 1
    print("经过%d次迭代后找到最终权向量求解值为:"%k,w.transpose(),"\n")
    return w
# 绘图函数
def Plot(A1,A2,w,ID,flag):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    '''(1)绘制散点图'''
    point1 = ax.scatter3D(A1[:,0].tolist(),A1[:,1].tolist(),A1[:,2].tolist(),c='red',marker = '+')  # type: ignore
    point2 = ax.scatter3D(A2[:,0].tolist(),A2[:,1].tolist(),A2[:,2].tolist(),c='yellow',marker = '+')  # type: ignore
    '''(2)绘制分类面'''
    '''
    三个特征的权向量格式为w=[w0,w1,w2,w3]
    则可得分类面方程为 : w1*x + w2*y + w3*z + w0 = 0
    推导可得 -w3*z = w1*x + w2*y + w0
    z = -(w1*x + w2*y + w0)/w3
    '''
    x = np.linspace(-1,8,10)
    y = np.linspace(-1,8,10)
    X,Y = np.meshgrid(x,y)
    w = w.A # 矩阵单个元素也属于一个独立矩阵，因此将matrix矩阵类型转为ndarray
    Z = -1*(w[1]*X+w[2]*Y+w[0])/w[3]
    ax.plot_surface(X,Y,Z,color='aqua')  # type: ignore
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    if(flag==True):
        str1 = "训练集中为第%d类"%ID[0]
        str2 = "训练集中为第%d类"%ID[1]
    else:
        str1 = "被判别为第%d类"%ID[0]
        str2 = "被判别为第%d类"%ID[1]
    plt.legend([point1,point2],[str1,str2])
# 测试集测试函数
def Test(B1,B2,del_line,ID,w,flag):
    true,false =0,0
    # 去除无用的数据
    B1 = np.delete(B1,del_line,axis=1)
    B2 = np.delete(B2,del_line,axis=1)
    # 样本增广化处理
    One = np.ones(25) # 1*25的行向量值均为1
    B1 = np.insert(B1,1,values=One,axis=1)
    B2 = np.insert(B2,1,values=One,axis=1)
    B = np.concatenate((B1,B2),axis=0) # B1与B2矩阵拼接在一起
    res1,res2 = [],[]
    for i in range(0,50):
        B_row = B[i] # 取出第i行
        id = B_row[0] # 取出测试集实际标号
        B_row = np.delete(B_row,0) # 矩阵中删除类别号
        y = np.dot(B_row,w)
        if y > 0:
            res1.append(B_row)
            Id = ID[0]
        else:
            res2.append(B_row)
            Id = ID[1]
        true,false = count_accuracy(true,false,id,Id)
    if flag==1:
        str='感知器批处理方法'
    elif flag==2:
        str='感知器单步处理方法'
    else:
        str='感知器最小平方误差判别方法'
    print("基于%s对第%d类和第%d类进行分类："%(str,ID[0],ID[1]))
    print("正确个数：",true)
    print("错误个数：",false)
    print("准确率：",true/(true+false),'\n')
    # 将分类结果转换为矩阵
    res1 = np.mat(res1)
    res2 = np.mat(res2)
    # 将第一列增广列删除
    res1 = np.delete(res1,0,axis=1)
    res2 = np.delete(res2,0,axis=1)
    return res1,res2

'''
1.读取训练集和测试集
注:在此处统计行数是为了兼容不同的样本集,
因为理论上说我们事先不会知晓有多少组数据
'''
f1 = open(r'F:\Code\PattenRecognization\Iris\3-Perceptron\test.txt') # 打开训练集
f2 = open(r'F:\Code\PattenRecognization\Iris\3-Perceptron\train.txt') # 打开测试集
lines1 = f1.readlines() # 把全部数据文件读到一个列表lines中
lines2 = f2.readlines()
Line1 = len(lines1) # 读取训练集行数
Line2 = len(lines2) # 读取训练集列数
A = Read(lines1,Line1,5)
B = Read(lines2,Line2,5)
'''
2.将三类样本拆分
'''
# 提取三类训练集
A1,A2,A3 = A[0:25],A[25:50],A[50:75]
# 提取三类测试集
B1,B2,B3 = B[0:25],B[25:50],B[50:75]
'''
3.利用批处理的梯度下降算法设计的感知器
A1--第一类 A2--另一类
del_line--要删除的列
ID--存储着两类的类别号
w--权向量 p--学习率 
'''
w = np.ones((4,1)) # 初始权向量
del_line = 4 # 我们选择前三个特征，因此第五列(第四个特征)删掉
p = 0.95 # 学习率
# 求解第一类和第二类的权向量w 
w1 = processing_batch(A1,A2,[1,2],del_line,w,p)
# 求解第一类和第三类的权向量w 
w2 = processing_batch(A1,A3,[1,3],del_line,w,p)
# 求解第二类和第三类的权向量w 
# w3 = processing_batch(A2,A3,[2,3],del_line,w,p)
print("在此省略，因为非线性无法迭代出一个权向量")

'''
4.绘制训练集的散点图以及分类面
flag为1则为训练集;flag为2则为测试集
'''
a = np.delete(A,[0,del_line],axis=1)
a1,a2,a3 = a[0:25],a[25:50],a[50:75] # 未增广的数据集
# 第一类和第二类的分类面以及训练集散点图
Plot(a1,a2,w1,[1,2],flag=1)
plt.title('第一类和第二类训练集的分布')
# 第一类和第三类的分类面以及训练集散点图
Plot(a1,a3,w2,[1,3],flag=1)
plt.title('第一类和第三类训练集的分布')
# 第二类和第三类的分类面以及训练集散点图
print("在此省略，因为非线性无法迭代出一个权向量")
'''
5.导入测试集对分类面进行检验
'''
# 检验第一类和第二类的准确率
res12,res21 = Test(B1,B2,del_line,[1,2],w1,1)
# 检验第一类和第三类的准确率
res13,res31 = Test(B1,B3,del_line,[1,3],w2,1)
'''
6.绘制测试集的散点图以及分类面
flag为1则为训练集;flag为2则为测试集
'''
# 第一类和第二类的分类面以及测试集集散点图
Plot(res12,res21,w1,[1,2],flag=0)
plt.title('第一类和第二类测试集的分布')
# 第一类和第三类的分类面以及测试集散点图
Plot(res13,res31,w2,[1,3],flag=0)
plt.title('第一类和第三类测试集的分布')
# 第二类和第三类的分类面以及测试集散点图
print("在此省略，因为非线性无法迭代出一个权向量")

plt.show()
