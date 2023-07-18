% % 为了程序的可复现性，后续程序使用存储下来的cbs_5000.mat
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 使用randn函数产生均值为0，方差σ^2 = 1，标准差σ=1的正态分布的随机矩阵，5000个数据，分两类。
clc;clear all;close all   % 清理
 
X = [randn(2500,2)+ones(2500,2);...       %续写下一行
     randn(2500,2)-ones(2500,2);];
X(1:2500,3)=1;
X(2501:5000,3)=2;
% % 原始样本数据二维分布
figure, plot(X(1:2500,1),X(1:2500,2),'ro')    %2500个总体第一类样本
 hold on,plot(X(2501:5000,1),X(2501:5000,2),'b*')  %2500个总体第二类样本
 grid;
 title('原始总体样本数据二维分布')
% % 初始的训练样本绘制图像
 figure, plot(X(1:2000,1),X(1:2000,2),'ro')    %取两千个训练集，1至2000
 hold on,plot(X(2501:4500,1),X(2501:4500,2),'b*')  %取两千个训练集，2501至4500
 grid;
 title('总体初始样本分布图')
 % % 原始测试样本分布图
 figure, plot(X(2001:2500,1),X(2001:2500,2),'ro')    %取五百个测试集
 hold on,plot(X(4501:5000,1),X(4501:5000,2),'b*')   %取五百个测试集
 grid;
 title('原始测试样本分布图')
 % % 存储工作区变量
 save ('cbs_5000.mat','X');
 
 % % 近邻法
biaoji=[];  %做一个空的标记数组用于后面比较类别
for i=2001:2500
    minlen=norm(X(1,1:2)-X(i,1:2))^2;  %二范数的平方,先选第一个训练点作为参考点
    for j=1:2000  %取第一类的样本先求距离
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=1;
        end
    end    %求出目前样本点和第一类每个点的最小距离
    for j=2501:4500  %取第一类的样本先求距离
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=2;
        end
    end    %求出目前样本点和第二类每个点的最小距离，此时可以知道样本点的类别了
    biaoji(i-2000)=flag;
end
% %同理，第二类测试点来求
for i=4501:5000
    minlen=norm(X(2501,1:2)-X(i,1:2))^2;  %二范数的平方,先选第一个训练点作为参考点
    for j=1:2000  %取第一类的样本先求距离
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=1;
        end
    end    %求出目前样本点和第一类每个点的最小距离
    for j=2501:4500  %取第一类的样本先求距离
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=2;
        end
    end    %求出目前样本点和第二类每个点的最小距离，此时可以知道样本点的类别了
    biaoji(i-4000)=flag;
end
% % 判断近邻法错误个数和错误率
num=0;
for i=1:500
    if biaoji(i)~=X(2001:2500,3)
        num=num+1;
    end
end
for i=1:500
    if biaoji(i)~=X(2001:2500,3)
        num=num+1;
    end
end
for i=501:1000
    if biaoji(i)~=X(4501:5000,3)
        num=num+1;
    end
end
 
% % 继前面后，剪辑近邻法(利用flag进行分类）
num1=0; 
for i=501:2000  %取1500个作为剪辑的考试集，500作为基本的考试集,共3000个参照集，1000参照集。
    minlen=norm(X(1,1:2)-X(i,1:2))^2;  %二范数的平方,先选第一个训练点作为参考点
    for j=1:500  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=1;
        end
    end
    for j=2501:3000  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=2;
        end
    end
    if X(i,3)~=flag
        X(i,3)=3;%用于后续筛选错误的训练样本出去
    end
end
 
for i=3001:4500  %取1500个作为剪辑的考试集，500作为基本的考试集,共3000个参照集，1000参照集。
    minlen=norm(X(1,1:2)-X(i,1:2))^2;  %二范数的平方,先选第一个训练点作为参考点
    for j=1:500  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=1;
        end
    end
    for j=2501:3000  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len
            minlen=len;  
            flag=2;
        end
    end
    if X(i,3)~=flag
        X(i,3)=3;  %用于后续筛选错误的训练样本出去，flag=3是错误的，等筛出去
    end
end
% % 剪辑后的训练集用于给测试集分类，并计算错误率和错误数量，这里无需给出剪辑后1000个测试样本的分布图，只需要得到错误了多少个，减少工作量
for i=2001:2500   %样本集 （这里开始用剪辑后的训练样本对测试样本进行最近邻法程序判别）
    minlen=norm(X(501,1:2)-X(i,1:2))^2;  %二范数的平方,先选第一个训练点作为参考点
    for j=501:2000  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3  %把之前剪辑的不正确的样本给淘汰出去
            minlen=len;  
            flag=1;
        end
    end
    for j=3001:4500  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3  %把之前剪辑的不正确的样本给淘汰出去
            minlen=len;  
            flag=2;
        end
    end
    if X(i,3)~=flag  %这些样本中不存在flag=3的
        num1=num1+1;
    end
end
for i=4501:5000 %样本集
    minlen=norm(X(501,1:2)-X(i,1:2))^2;  %二范数的平方,先选第一个训练点作为参考点
    for j=501:2000  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3  %把之前剪辑的不正确的样本给淘汰出去
            minlen=len;  
            flag=1;
        end
    end
    for j=3001:4500  
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3  %把之前剪辑的不正确的样本给淘汰出去
            minlen=len;  
            flag=2;
        end
    end
    if X(i,3)~=flag  %这些样本中不存在flag=3的
        num1=num1+1;
    end
end
 
% % 绘制剪辑近邻法后的测试样本点(样本分布图）， 样本分布是用来对测试集进行分类的参考，这里小于3000
n=0;  %统计剪辑后，还剩下多少个考试集（测试样本）
figure;
grid;
%第一类样本
for i=501:2000  %取原考试集中余下的NTE样本
    if (X(i,3)~=3)
        plot(X(i,1),X(i,2),'ro');
        hold on;
        n=n+1;
    end
end
%第二类样本
for i=3001:4500  %取原考试集中余下的NTE样本
    if (X(i,3)~=3)
        plot(X(i,1),X(i,2),'b*');
        hold on;
        n=n+1;
    end
end
title('剪辑后的样本分布图')
 
% % 压缩近邻法
%利用if-else把第一个样本找出来，即选取上面剪辑过后的第一个测试样本作为Xs集，这里两个，每类各取一个
for i=501:2000
    if X(i,3)~=3
        X(i,4)=1;  %定义一个新的列来确定Xs和Xg两个类别的样本
        break;   %取一个跳出for循环
    end
end
for i=3001:4500
    if X(i,3)~=3
        X(i,4)=1;  %定义一个新的列来确定Xs和Xg两个类别的样本
        break;   %取一个跳出for循环
    end
end
% % 进行Xs和Xg两个集合直接的转换，最终方便留下Xs作为待测样本的训练样本
xflag=1; %启动循环的条件
while xflag==1  %判断要不要终止Xs与Xg转移样本的过程
    xflag=0; %假设只有这一次循环就结束，后面再改
    for i=501:2000  %上面剪辑剩下的当成待测样本
        flag=1; %定义初始量假设是第一类
        if X(i,3)~=3 && X(i,4)~=1   %选Xg里面的样本
            minlen=norm(X(501,1:2)-X(i,1:2))^2;   %Xs对Xg使用近邻法看看能不能正确分类
            for j=501:2000
                len=norm(X(i,1:2)-X(j,1:2))^2;
                if minlen>len && X(j,3)~=3 && X(j,4)==1      %把之前剪辑的不正确的样本给淘汰出去，并取Xs的
                    minlen=len;  
                    flag=1;
                end
            end
            for j=3001:4500
                len=norm(X(i,1:2)-X(j,1:2))^2;
                if minlen>len && X(j,3)~=3 && X(j,4)==1      
                    minlen=len;  
                    flag=2;
                end
            end
            if X(i,3)~=flag  %分类错误的
                X(i,4)=1; %放到Xs中
                xflag=1; %说明还得再循环
            end
        end
    end
     for i=3001:4500  %上面剪辑剩下的当成待测样本
        flag=2;  %重置假设是第二类
        if X(i,3)~=3 && X(i,4)~=1   %选Xg里面的样本
            minlen=norm(X(501,1:2)-X(i,1:2))^2;   %Xs对Xg使用近邻法看看能不能正确分类
            for j=501:2000
                len=norm(X(i,1:2)-X(j,1:2))^2;
                if minlen>=len && X(j,3)~=3 && X(j,4)==1      %把之前剪辑的不正确的样本给淘汰出去，并取Xs的
                    minlen=len;  
                    flag=1;
                end
            end
            for j=3001:4500
                len=norm(X(i,1:2)-X(j,1:2))^2;
                if minlen>=len && X(j,3)~=3 && X(j,4)==1      
                    minlen=len;  
                    flag=2;
                end
            end
            if X(i,3)~=flag  %分类错误的
                X(i,4)=1; %放到Xs中
                xflag=1; %说明还得再循环,知道xflag=0时退出，这时Xs真正分出来了
            end
        end
     end
end
 
% % 用压缩后的样本当成训练样本，对待测样本进行近邻法分析，得出错误个数和错误率
num2=0;
for i=2001:2500
    minlen=norm(X(501,1:2)-X(i,1:2))^2; 
    for j=501:2000
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3 && X(j,4)==1   %选择Xs中的作为训练样本对待测样本进行分类
            minlen=len;
            flag=1;
        end
    end
    for j=3001:4500
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3 && X(j,4)==1
            minlen=len;
            flag=2;
        end
    end
    if X(i,3)~=flag;
        num2=num2+1;
    end
end
for i=4501:5000
    minlen=norm(X(501,1:2)-X(i,1:2))^2; 
    for j=501:2000
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3 && X(j,4)==1
            minlen=len;
            flag=1;
        end
    end
    for j=3001:4500
        len=norm(X(i,1:2)-X(j,1:2))^2;
        if minlen>len && X(j,3)~=3 && X(j,4)==1
            minlen=len;
            flag=2;
        end
    end
    if X(i,3)~=flag;
        num2=num2+1;
    end
end
 
% % 绘制剪辑、压缩后的训练样本图像
t=0; %统计还有多少个样本点
figure;
grid;
for i=501:2000
    if X(i,3)~=3 && X(i,4)==1
        plot(X(i,1),X(i,2),'ro');
        hold on;
        t=t+1;
    end
end
for i=3001:4500
    if X(i,3)~=3 && X(i,4)==1
        plot(X(i,1),X(i,2),'b*');
        hold on;
        t=t+1;
    end
end
title('压缩后的训练样本图像')
