#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# 将列表中的数据切片读入矩阵
def Read(lines,m,n):
    A = np.zeros((m, n))
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        A[A_row:] = list[0:5]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    return A
# 输出权向量
def show_w(w):
    print("第一类:",w[0].transpose())
    print("第二类:",w[1].transpose())
    print("第三类:",w[2].transpose())
# 计算准确率
def count_accuracy(true,false,id,Id):
    if id==Id:
        true+=1
    else:
        false+=1
    return true,false
# 对样本的训练函数
'''
对样本的训练函数
A--某一类的训练样本
w--存储着三类目前的权向量，为列表[w1,w2,w3]
right_id,该样本正确的id值
count--数据集里连续分类正确的个数
k--迭代次数 p--学习率
'''
def Train(A,w,right_id,count,k,p,i,max_k,Error):
    y = [np.dot(A[i,:],w[0]),np.dot(A[i,:],w[1]),np.dot(A[i,:],w[2])]
    # print("aaaaaaaaaaaaaaaaaaaaaaa",y)
    # 寻找最大值的下角标
    cur_id = y.index(max(y))
    # 如果此时最大值的下角标有cur_id + 1 = right_id，即判别类别与实际类别相同
    if cur_id + 1 == right_id:
        count = count + 1
        # print("第%d次迭代中,该权向量对目前所检验的样本分类正确"%k)
        # print("三类权向量值依旧为")
    # 如果此时分类错误，但是迭代次数足够多且两个样本分类函数值之间差小于0.2，我们也认为是正确的
    # （因为第二类和第三类是明显的非线性，无法通过线性判别求解，允许少量误差的存在）
    elif k>=max_k and abs(y[1]-y[2]) < Error:
        count = count + 1
        # print("第%d次迭代中,虽然分类错误，但是误差已足够小，因此也认为正确"%k)
        # print("三类权向量值依旧为")
    # 其他情况只能认为是分类错误
    else:
        temp = p * np.mat(A[i,:]).transpose() 
        w[cur_id] = w[cur_id] - temp
        # 因为y[right_id-1]不是最大的，所以需要增加w[right_id-1]
        w[right_id - 1] = w[right_id - 1] + temp
        count = 0 # 分类错误，count重置
        # print("第%d次迭代中,该权向量对目前所检验的样本错分成第%d类"%(k,cur_id+1))
        # print("三类权向量值更新为")
    # show_w(w)
    return w,count

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
# A2 = A3

'''
3.去除无用的数据
例如目前需要绘制第一类和第二类的分类面，选取前三个特征进行分类
就需要去除第一列（标签）和第五列的数据（第四个特征）
'''
del_line = 1
A1 = np.delete(A1,[0,del_line],axis=1)
A2 = np.delete(A2,[0,del_line],axis=1)
A3 = np.delete(A3,[0,del_line],axis=1)
a1,a2,a3 = A1,A2,A3 # 寄存后续绘制散点图使用

'''
4.样本增广化处理
'''
One = np.ones(25) # 1*25的行向量值均为1
A1 = np.insert(A1,0,values=One,axis=1) # 在第0列插入One向量（变成了列向量）
A2 = np.insert(A2,0,values=One,axis=1)
A3 = np.insert(A3,0,values=One,axis=1)


'''
5.采用单步处理的梯度下降算法进行迭代运算
计算权向量w和迭代次数k
w为4*1的向量,初始化时设定为[1,1,1,1]^T
设定学习率p
'''
w = [np.ones((4,1)),np.ones((4,1)),np.ones((4,1))]
k = 1 # 迭代次数
p = 0.1 # 学习率
count = 0 # 记录数据集里连续分类正确的个数，count=75代表全部分类成功
i = 0 # 目前在用第i个数据检验正确性
max_k = 10000 # 我们认为足够大的迭代次数，为了应对非线性关系的存在而设置
Error = 1.5 # 非线性关系中分类允许的感知器误差
# '''开始迭代！！'''
'''思路：为了使三类样本均受到相同周期的训练，
在迭代时每一个while循环均训练三个样本,
而这三个样本就是从三类中分别取出一个进行训练
避免一口气训练完成第一类样本后，结果对二三类样本严重违和
保证三类的权向量变化幅度相对较小，更快收敛到正确值。'''
while 1:
    # 对第一类样本的第i+1组数据训练
    w,count = Train(A1,w,1,count,k,p,i,max_k,Error)
    k = k + 1
    # print("count:",count)
    '''判断是否迭代完毕'''
    if count>=75: # 连续75个样本分类正确
        # print("迭代完毕！")
        break
    # 对第二类样本的第i+1组数据训练
    w,count = Train(A2,w,2,count,k,p,i,max_k,Error)
    k = k + 1
    # print("count:",count)
    '''判断是否迭代完毕'''
    if count>=75: # 连续75个样本分类正确
        # print("迭代完毕！")
        break
    # 对第三类样本的第i+1组数据训练
    w,count = Train(A3,w,3,count,k,p,i,max_k,Error)
    k = k + 1
    # print("count:",count)
    '''判断是否迭代完毕'''
    if count>=75: # 连续75个样本分类正确
        # print("迭代完毕！")
        break
    '''下一轮的迭代'''
    i = (i + 1) % 25
print("经过%d次迭代后找到最终权向量求解值为:"%k)
show_w(w)

'''
6.导入测试集对分类面进行检验
'''
true,false =0,0
# 去除无用的数据
B = np.delete(B,del_line,axis=1)
# 样本增广化处理
One = np.ones(75) # 1*75的行向量值均为1
B = np.insert(B,1,values=One,axis=1)
# res1,res2,res3 = [],[],[]
res = [[],[],[]]
for i in range(0,75):
    B_row = B[i] # 取出第i行
    id = B_row[0] # 取出测试集实际标号
    B_row = np.delete(B_row,0) # 矩阵中删除类别号
    y = [np.dot(B_row,w[0]),np.dot(B_row,w[1]),np.dot(B_row,w[2])]
    # 最大值的下角标+1就是我们测试估计出的类别号
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",y)
    Id = y.index(max(y)) + 1
    # 放入对应类别的列表
    res[Id - 1].append(B_row)
    true,false = count_accuracy(true,false,id,Id)
print("基于多类分类器对三类样本进行分类：")
print("正确个数：",true)
print("错误个数：",false)
print("准确率：",true/(true+false),'\n')


'''
7.绘制测试集的散点图以及分类面
'''
# 将分类结果转换为矩阵
res1,res2,res3 = np.mat(res[0]),np.mat(res[1]),np.mat(res[2])
# 将第一列增广列删除
res1,res2,res3 = np.delete(res1,0,axis=1),np.delete(res2,0,axis=1),np.delete(res3,0,axis=1)
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
'''(1)绘制散点图'''
point1 = ax.scatter3D(res1[:,0].tolist(),res1[:,1].tolist(),res1[:,2].tolist(),c='red',marker = '+')  # type: ignore
point2 = ax.scatter3D(res2[:,0].tolist(),res2[:,1].tolist(),res2[:,2].tolist(),c='yellow',marker = '+')  # type: ignore
point3 = ax.scatter3D(res3[:,0].tolist(),res3[:,1].tolist(),res3[:,2].tolist(),c='blue',marker = '+')  # type: ignore
plt.legend([point1,point2,point3],["测试集中判定为第1类","测试集中判定为第2类","测试集中判定为第3类"])
'''(2)绘制分类面
三个特征的权向量格式为w=[w0,w1,w2,w3]
则可得分类面方程为 : w1*x + w2*y + w3*z + w0 = 0
推导可得 -w3*z = w1*x + w2*y + w0
z = -(w1*x + w2*y + w0)/w3
'''
x = np.linspace(-1,8,3)
y = np.linspace(-1,8,3)
X,Y = np.meshgrid(x,y)
# 矩阵单个元素也属于一个独立矩阵，因此将matrix矩阵类型转为ndarray
w1,w2,w3 = w[0].A,w[1].A,w[2].A # type: ignore 
Z1 = -1*(w1[1]*X+w1[2]*Y+w1[0])/w1[3] # 第一类分类面
Z2 = -1*(w2[1]*X+w2[2]*Y+w2[0])/w2[3] # 第二类分类面
Z3 = -1*(w3[1]*X+w3[2]*Y+w3[0])/w3[3] # 第三类分类面
ax.plot_surface(X,Y,Z1,color='mistyrose')  # type: ignore
ax.plot_surface(X,Y,Z2,color='lightyellow')  # type: ignore
ax.plot_surface(X,Y,Z3,color='cyan')  # type: ignore

'''
8.因分类面相交容易将点遮住，现单独输出三类样本的分类面
'''
# 第一类
fig = plt.figure()
ax = plt.axes(projection='3d')
point1 = ax.scatter3D(res1[:,0].tolist(),res1[:,1].tolist(),res1[:,2].tolist(),c='red',marker = '+')  # type: ignore
point2 = ax.scatter3D(res2[:,0].tolist(),res2[:,1].tolist(),res2[:,2].tolist(),c='yellow',marker = '+')  # type: ignore
point3 = ax.scatter3D(res3[:,0].tolist(),res3[:,1].tolist(),res3[:,2].tolist(),c='blue',marker = '+')  # type: ignore
plt.legend([point1,point2,point3],["测试集中判定为第1类","测试集中判定为第2类","测试集中判定为第3类"])
Z1 = -1*(w1[1]*X+w1[2]*Y+w1[0])/w1[3] # 第一类分类面
ax.plot_surface(X,Y,Z1,color='red')  # type: ignore

# 第二类
fig = plt.figure()
ax = plt.axes(projection='3d')
point1 = ax.scatter3D(res1[:,0].tolist(),res1[:,1].tolist(),res1[:,2].tolist(),c='red',marker = '+')  # type: ignore
point2 = ax.scatter3D(res2[:,0].tolist(),res2[:,1].tolist(),res2[:,2].tolist(),c='yellow',marker = '+')  # type: ignore
point3 = ax.scatter3D(res3[:,0].tolist(),res3[:,1].tolist(),res3[:,2].tolist(),c='blue',marker = '+')  # type: ignore
plt.legend([point1,point2,point3],["测试集中判定为第1类","测试集中判定为第2类","测试集中判定为第3类"])
Z2 = -1*(w2[1]*X+w2[2]*Y+w2[0])/w2[3] # 第二类分类面
ax.plot_surface(X,Y,Z2,color='yellow')  # type: ignore

# 第三类
fig = plt.figure()
ax = plt.axes(projection='3d')
point1 = ax.scatter3D(res1[:,0].tolist(),res1[:,1].tolist(),res1[:,2].tolist(),c='red',marker = '+')  # type: ignore
point2 = ax.scatter3D(res2[:,0].tolist(),res2[:,1].tolist(),res2[:,2].tolist(),c='yellow',marker = '+')  # type: ignore
point3 = ax.scatter3D(res3[:,0].tolist(),res3[:,1].tolist(),res3[:,2].tolist(),c='blue',marker = '+')  # type: ignore
plt.legend([point1,point2,point3],["测试集中判定为第1类","测试集中判定为第2类","测试集中判定为第3类"])
Z3 = -1*(w3[1]*X+w3[2]*Y+w3[0])/w3[3] # 第三类分类面
ax.plot_surface(X,Y,Z3,color='blue')  # type: ignore

plt.show()