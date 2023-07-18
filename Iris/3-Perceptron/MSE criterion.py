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
# 基于MSE设计的感知器(伪逆矩阵法)
def MSE_pinv(A1,A2,ID,del_line,w,b):
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
    '''
    (3)采用最小平方误差判别进行运算(伪逆矩阵法)
    '''
    w = np.linalg.pinv(A).dot(b)
    print("采用伪逆矩阵法对第%d类与第%d类分类找到最终权向量求解值为:"%(ID[0],ID[1]),w.transpose())
    return w

# 基于MSE设计的感知器(单样本修正法/LMS):
def LMS(A1,A2,ID,del_line,w,b,p,n_interations):
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
    A = np.mat(A)
    '''
    (3)采用最小平方误差判别进行运算(单样本修正法LMS)
    ''' 
    k = 0 # 迭代次数
    i = 0 # 目前在用第i个数据检验正确性
    '''开始迭代！！'''
    for k in range(n_interations):
    # while 1:
        gradients = np.dot(A[i,:].transpose(),np.dot(A[i,:],w)-b[i]) # 计算样本梯度
        J = p * gradients # 平方误差作为惩罚
        w = w - J
        # print("第%d次迭代中,求解惩罚值为%lf"%(k+1,J))
        # print("更新权向量值为:",w.transpose())
        Error = np.linalg.norm(J)
        if Error<= 1e-10:
            break
        '''下一轮的迭代'''
        i = (i + 1) % 50
        k = k+1
    print("采用单样本修正法对第%d类与第%d类分类,经过%d次迭代后找到最终权向量求解值为:"%(ID[0],ID[1],k+1),w.transpose())
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
        str='逆矩阵法MSE'
    elif flag==2:
        str='梯度下降法MSE'
    else:
        str='LMS最小均方根算法'
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
3.利用基于MSE或LMS设计的感知器
我们通过前面的实验设计的感知器可以发现
只有第二类和第三类进行分类时，利用常规的梯度下降是无法收敛出一个w的
因此本代码里只对第二类与第三类进行基于最小平方误差准则的感知器求解
'''
w = 0.01*np.ones((4,1)) # 初始权向量
del_line = 4 # 我们选择前三个特征，因此第五列(第四个特征)删掉
b = 0.001*np.mat(np.ones((50,1)))
# 采用伪逆矩阵法求解第二类和第三类的权向量w 
w_pinv = MSE_pinv(A2,A3,[2,3],del_line,w,b)
# 采用单样本修正法求解第二类和第三类的权向量w 
p = 0.001 # 学习率
n_interations = 93000 # 迭代次数
w_lms = LMS(A2,A3,[2,3],del_line,w,b,p,n_interations)

'''
4.绘制基于MSE设计的感知器训练集的散点图以及分类面
flag为1则为训练集;flag为2则为测试集
'''
a = np.delete(A,[0,del_line],axis=1)
a1,a2,a3 = a[0:25],a[25:50],a[50:75] # 未增广的数据集
# 基于伪逆矩阵法的MSE设计的第二类和第三类的分类面以及训练集散点图
Plot(a2,a3,w_pinv,[2,3],flag=1)
plt.title('基于伪逆矩阵法的MSE设计的第二类和第三类分类面以及训练集的分布')
# 基于单样本修正法的LMS设计的第二类和第三类的分类面以及训练集散点图
Plot(a2,a3,w_lms,[2,3],flag=1)
plt.title('基于LMS设计的第二类和第三类分类面以及训练集的分布')
'''
5.导入测试集对分类面进行检验
'''
# 检验伪逆矩阵法的MSE对第二类和第三类分类的准确率
res_pinv1,res_pinv2 = Test(B2,B3,del_line,[2,3],w_pinv,1)
# 检验单样本修正法的LMS对第二类和第三类分类的准确率
res_lms1,res_lms2 = Test(B2,B3,del_line,[2,3],w_lms,3)
'''
6.绘制测试集的散点图以及分类面
flag为1则为训练集;flag为2则为测试集
'''
# 采用伪逆矩阵法的MSE对第二类和第三类的分类面以及测试集散点图
Plot(res_pinv1,res_pinv2,w_pinv,[2,3],flag=2)
plt.title('采用伪逆矩阵法的MSE对第二类和第三类测试结果的分布')
# 采用单样本修正法的LMS对第二类和第三类的分类面以及测试集散点图
Plot(res_pinv1,res_pinv2,w_lms,[2,3],flag=2)
plt.title('采用单样本修正法的LMS对第二类和第三类测试结果的分布')

plt.show()
