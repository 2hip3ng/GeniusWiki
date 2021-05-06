人工智能体系的基础知识涵盖许多交叉学科信息，比如数学、计算机科学与技术、通信工程等。本文介绍与信息论（典型通信工程学科）、概率论中一些与人工智能相关的常用概念介绍，帮助理解人工智能体系。

# 1. 几个概念

设$X$是一个取有限个值的离散随机变量，其概率分布为：
$$
P(X=x_i) = p_i,  \  i =1,2,...,n \tag{1}
$$
设有随机变量$(X,Y)$，其联合概率分布为：
$$
p(X=x_i, Y=y_j) = p_{ij}, \ i = 1,2, ...,n; j = 1,2,...,m \tag{2}
$$


1. 熵(Entropy)：表示了随机变量不确定的度量，定义为：

$$
H(X) = -\sum_{i=1}^{n}p_i log \ p_i \tag{3}
$$

​        熵越大，随机变量的不确定性就越大。



2. KL散度（KL divergence, Kullback-Leibler divergence）：描述两个概率分布$Q(x)$与$P(x)$相似度的一种度量，记作$D(Q||P)$。对离散随机变量，KL散度定义为：

$$
D(Q||P) = \sum_{i}Q(x_i)log\frac{Q(x_i)}{P(x_i)} \tag{4}
$$

​       对连续随机变量，KL散度定义为：
$$
D(Q||P) = \int Q(x)log \frac{Q(x)}{P(x)} dx \tag{5}
$$
​		KL散度是非对称的，$D(Q||P) != D(P||Q)$，不是严格的距离度量。$D(Q||P)\ge 0$ （挖坑待证明），当且仅当$Q=P$，$D(Q||P)=0$。



3. 交叉熵（Cross Entropy），定义为：

$$
C(Q, P) = - \sum_{i} Q(x_i) log P(x_i) \tag{6}
$$

4. KL散度与交叉熵之间的关系：

$$
D(Q||P) = \sum_{i}Q(x_i)log\frac{Q(x_i)}{P(x_i)} \\
D(Q||P) = \sum_{i}Q(x_i)logQ(x_i)-Q(x_i)logP(x_i) \tag{7}
$$

​		将公式（3）及公式（6）代入公式（7）得，
$$
D(Q||P) =  C(Q,P)  - H(Q) \tag{8}
$$

$$
KL散度 = 交叉熵 - 熵 \tag{9}
$$

​		在机器学习问题中，$Q(x)$数据标签的真实分布，$H(Q)$为固定值，最小化KL散度等价于最小化交叉熵。



5. 条件熵：



6. 信息增益





极大似然

交叉熵与极大似然

基尼指数



# 2. 参考文献

1. 统计学习方法（第二版），李航

2. https://blog.csdn.net/Dby_freedom/article/details/83374650?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&dist_request_id=1331988.12589.16188303449515571&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control

