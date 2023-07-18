import pandas as pd
import numpy as np
import math
# 读入样本数据
data = pd.read_excel('酒瓶分类.xlsx')
# 拆分训练集和样本集
data_tr = data[0:29]
data_te = data[29:]
# 初始化
N = len(data_tr)                           # N个测试样本
w = len(set(data.iloc[:,4]))               # w个类别
n = 3                                     # n个特征
N1 = list(data_tr['所属类别']).count(1)        # 测试样本中第一类的数量  
N2 = list(data_tr['所属类别']).count(2)        # 测试样本中第二类的数量  
N3 = list(data_tr['所属类别']).count(3)        # 测试样本中第三类的数量  
N4 = list(data_tr['所属类别']).count(4)        # 测试样本中第四类的数量  
A = data.iloc[:N1,1:4]                     # A belongs to w1
B = data.iloc[N1:N2+N1,1:4]                # B belongs to w2
C = data.iloc[N2+N1:N1+N2+N3,1:4]          # C belongs to w3
D = data.iloc[N1+N2+N3:N1+N2+N3+N4,1:4]    # D belongs to w4
# 先验概率
pw1 = N1/N
pw2 = N2/N
pw3 = N3/N
pw4 = N4/N
def get_p(A,pw):
    P_ls = []
    x1 = np.array(A.mean())                 # 求样本均值
    s1 = np.mat(A.cov())                    # 求样本协方差矩阵
    s1_ = s1.I                               # 求协方差矩阵的逆矩阵
    s11 = np.linalg.det(s1)                  # 求协方差矩阵的行列式
    for i in range(30):
        u = np.mat(data_te.iloc[i,1:4]-x1)
        P1=-1/2*u*s1_*u.T+math.log(pw1)-1/2*math.log(s11)
        P_ls.append(P1)
    return P_ls
cnt = 0
P1 = get_p(A,pw1)
P2 = get_p(B,pw2)
P3 = get_p(C,pw3)
P4 = get_p(D,pw4)
for i in range(30):
    P = [P1[i],P2[i],P3[i],P4[i]]
    data.iloc[i+29,5] = P.index(max(P))+1
    if data.iloc[i+29,5] == data.iloc[i+29,4]:
        cnt += 1
accuracy = cnt/len(data_te)
data.to_excel('result_new.xlsx',index = None)