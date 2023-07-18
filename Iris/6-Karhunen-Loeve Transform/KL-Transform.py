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
# 数据集初始化
def DataInit(): 
    '''1.读取训练集和测试集'''
    f1,f2 = open('train.txt'),open('test.txt') # 打开训练集和测试集
    lines1,lines2 = f1.readlines(),f2.readlines() # 把全部数据文件读到一个列表lines中
    Line1,Line2 = len(lines1),len(lines2) # 读取训练集合测试集的行数
    A,B = Read(lines1,Line1,5),Read(lines2,Line2,5)
    return A,B
# KL变换————使用总类内离散度矩阵作为产生矩阵
def KL_Transform(data1,data2,data3):
    '''1.计算三类均值向量'''
    mean1 = np.mean(data1[:,1:],axis=0)
    mean2 = np.mean(data2[:,1:],axis=0)
    mean3 = np.mean(data3[:,1:],axis=0)

    '''2.计算总类内离散度矩阵'''
    s1 = data1[:,1:]-mean1
    s1 = np.dot(s1.transpose(),s1)
    s2 = data2[:,1:]-mean2
    s2 = np.dot(s2.transpose(),s2)
    s3 = data3[:,1:]-mean3
    s3 = np.dot(s3.transpose(),s3)
    s = s1+s2+s3
    print(s)

    '''3.计算总类内离散度矩阵的特征值、特征向量'''
    val,vec = np.linalg.eig(s) # 特征值与特征向量(原理：AX=λX 即 |R-λI|=0 )
    print(val)
    print(vec)

    '''4.可视化各特征成分所占比例'''
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    x = ["特征1","特征2","特征3","特征4"]
    y = val
    colors = [ 'gold', 'lightskyblue','yellowgreen', 'lightcoral']
    plt.pie(y,labels=x,colors=colors,autopct='%1.1f%%',shadow=False,startangle=90)
    plt.axis('equal') # 显示为圆（避免比例压缩为椭圆）
    plt.title('各特征成分所占比例')

    '''5.降维打击:四维降到三维'''
    index = np.argsort(-val) # 降序排序返回下角标
    index = index[:3] # 取出前三大特征
    data = np.concatenate((data1,data2,data3),axis=0) # 数据集拼接
    new_data = np.dot(data[:,1:],vec[index].transpose())
    new_data = np.concatenate((np.mat(data[:,0]).T,new_data),axis=1) # 插入标签
    new_data1 = new_data[:50,:]
    new_data2 = new_data[50:100,:]
    new_data3 = new_data[100:,:]
    return new_data1,new_data2,new_data3

# 基于MSE设计的感知器(伪逆矩阵法)
def MSE_pinv(A1,A2,ID,w,b):
    '''1.去除无用的数据'''
    A1 = np.delete(A1,0,axis=1)
    A2 = np.delete(A2,0,axis=1)

    '''2.样本增广化、规范化处理'''
    # 样本增广化
    One = np.ones(25) # 1*25的行向量值均为1
    A1 = np.insert(A1,0,values=One,axis=1) # 在第0列插入One向量（变成了列向量）
    A2 = np.insert(A2,0,values=One,axis=1)
    # 样本规范化
    A = np.concatenate((A1,-1*A2),axis=0) # A1与A2矩阵拼接在一起

    '''3.采用最小平方误差判别进行运算(伪逆矩阵法)'''
    w = np.linalg.pinv(A).dot(b)
    print("采用伪逆矩阵法对第%d类与第%d类分类找到最终权向量求解值为:"%(ID[0],ID[1]),w.transpose())
    return w
# 基于MSE设计的感知器(单样本修正法/LMS):
def LMS(A1,A2,ID,w,b,p,n_interations):
    '''(1)去除无用的数据'''
    A1 = np.delete(A1,0,axis=1)
    A2 = np.delete(A2,0,axis=1)
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
        if Error<= 1e-11:
            break
        '''下一轮的迭代'''
        i = (i + 1) % 50
        k = k+1
    print("采用单样本修正法对第%d类与第%d类分类,经过%d次迭代后找到最终权向量求解值为:"%(ID[0],ID[1],k+1),w.transpose())
    return w
# 计算准确率
def count_accuracy(true,false,id,Id):
    if id==Id:
        true+=1
    else:
        false+=1
    return true,false
# 测试集测试函数
def Test(B1,B2,ID,w,flag):
    true,false =0,0
    # 样本增广化处理
    One = np.ones(25) # 1*25的行向量值均为1
    B1 = np.insert(B1,1,values=One,axis=1)
    B2 = np.insert(B2,1,values=One,axis=1)
    B = np.concatenate((B1,B2),axis=0) # B1与B2矩阵拼接在一起
    res1,res2 = [],[]
    for i in range(0,50):
        B_row = B[i] # 取出第i行
        
        id = B_row[0,0] # 取出测试集实际标号
        # print(id)
        B_row = np.delete(B_row,0) # 矩阵中删除类别号
        # print(B_row)
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
    else:
        str='LMS最小均方根算法'
    print("基于%s对第%d类和第%d类进行分类："%(str,ID[0],ID[1]))
    print("正确个数：",true)
    print("错误个数：",false)
    print("准确率：",true/(true+false),'\n')
   
    # 将第一列增广列删除
    res3,res4 = [],[]
    for item in res1:
        res3.append([item[0,1], item[0,2],item[0,3]])
    for item in res2:
        res4.append([item[0,1], item[0,2],item[0,3]])
    res3,res4 = np.mat(res3),np.mat(res4) # 列表转矩阵

    return res3,res4
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


'''一、数据集初始化'''
A,B = DataInit()
id = np.mat(A[:,0]).transpose() # 记录样本集的标号(75*1列向量)
A1,A2,A3 = A[0:25],A[25:50],A[50:75] # 提取三类训练集
B1,B2,B3 = B[0:25],B[25:50],B[50:75] # 提取三类测试集
data1 = np.concatenate((A1,B1),axis=0) # 各类训练集测试集拼接
data2 = np.concatenate((A2,B2),axis=0)
data3 = np.concatenate((A3,B3),axis=0)

data = np.concatenate((A,B),axis=0) # 训练集测试集拼接
data = np.delete(data,0,axis=1) # 删除第一列(类别号)

'''二、采用KL变换进行数据降维'''
data1,data2,data3 = KL_Transform(data1,data2,data3) # 分别为第123类的训练集+测试集
A1,B1 = data1[:25,:],data1[25:,:] # 降维后的第一类训练集、测试集
A2,B2 = data2[:25,:],data2[25:,:] # .........二类.............
A3,B3 = data3[:25,:],data3[25:,:] # .........三类.............

'''三、绘制训练集散点图'''
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
point1 = ax.scatter3D(A1[:,1].tolist(),A1[:,2].tolist(),A1[:,3].tolist(),c='red',marker = '+')  # type: ignore
point2 = ax.scatter3D(A2[:,1].tolist(),A2[:,2].tolist(),A2[:,3].tolist(),c='yellow',marker = '+')  # type: ignore
point3 = ax.scatter3D(A3[:,1].tolist(),A3[:,2].tolist(),A3[:,3].tolist(),c='blue',marker = '+') # type: ignore
ax.set(xlabel="降维后特征一", ylabel="降维后特征二", zlabel="降维后特征三")
plt.legend([point1,point2,point3],["第一类","第二类","第三类"])

'''四、最小平方误差判别'''
w = 0.01*np.ones((4,1)) # 初始权向量
b = 0.001*np.mat(np.ones((50,1)))
# 采用伪逆矩阵法求解第二类和第三类的权向量w 
w_pinv = MSE_pinv(A2,A3,[2,3],w,b)
# 采用单样本修正法求解第二类和第三类的权向量w 
p = 0.001 # 学习率
n_interations = 85000 # 迭代次数
w_lms = LMS(A2,A3,[2,3],w,b,p,n_interations)

'''五、导入测试集对分类面进行检验'''
# 检验伪逆矩阵法的MSE对第二类和第三类分类的准确率
res_pinv1,res_pinv2 = Test(B2,B3,[2,3],w_pinv,1)
# 检验单样本修正法的LMS对第二类和第三类分类的准确率
res_lms1,res_lms2 = Test(B2,B3,[2,3],w_lms,3)


'''六、绘制分类后的散点图以及分类面'''
# 采用伪逆矩阵法的MSE对第二类和第三类的分类面以及测试集散点图
Plot(res_pinv1,res_pinv2,w_pinv,[2,3],flag=2)
plt.title('采用伪逆矩阵法的MSE对第二类和第三类测试结果的分布')
# 采用单样本修正法的LMS对第二类和第三类的分类面以及测试集散点图
Plot(res_pinv1,res_pinv2,w_lms,[2,3],flag=2)
plt.title('采用单样本修正法的LMS对第二类和第三类测试结果的分布')











plt.show()