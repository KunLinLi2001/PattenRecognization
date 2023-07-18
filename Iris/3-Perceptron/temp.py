# 导包
import  numpy as np
import matplotlib.pyplot as plt

# 构造数据，模拟100行1的数据
X = 2* np.random.rand(100,1)
y = 4+3*X +np.random.randn(100,1)

# 将X于1列100行连接
X_b = np.c_[np.ones((100,1)),X]

# print(X_b)

# 学习率阿尔法的设定
rate = 0.01

# 设置迭代次数
n_interations = 10000
m = 100

#第一步：初始化
theta = np.random.randn(2,1)
count = 0

# 第二步：代入公式
for interation in range(n_interations):
    count += 1
    index = np.random.randint(m)
    # 切片
    Xi = X_b[index:index+1]
    yi = y[index:index+1]
    gradients = Xi.T.dot(Xi.dot(theta)-yi)
    theta = theta - rate*gradients
# print(count)
print(theta) # 梯度下降法求得的参数值
print(np.linalg.pinv(X_b).dot(y)) # 伪逆法求得的参数值

# 绘图准备
X_new = np.array([[0],[2]])
X_new_b = np.c_[(np.ones((2,1))),X_new]
# print(X_new_b)
y_predict = X_new_b.dot(theta)
# print(y_predict)

# 绘图模块
plt.plot(X_new,y_predict,'r-')
plt.plot(X,y,'b.')
plt.title('MSE');
plt.xlabel('X');
plt.ylabel('Y');
plt.axis([0,2,0,15])
plt.show()
