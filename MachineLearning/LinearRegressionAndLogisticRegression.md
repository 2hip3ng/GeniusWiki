回归人类最基础判断事物本质方法。

# 1. 线性回归（Linear Regression）

## 1.1 引言

线性回归，构造一条直线来拟合带有近线性规律的数据。如:

<img src="../images/Linear-Regression-1.png" alt="线性回归样例" width="300" height="300" />

## 1.2 模型

根据直线方程，可将线性回归表示为：
$$
h(x) = w_1x^{(1)}+w_2x^{(2)} + ... + w_nx^{(n)} + b  \tag1
$$
其中：

* $x$为输入数据
* $x^{(1)}$、$x^{(2)}$、$x^{(3)}$ 、...、$x^{(n)}$为输入数据的各列属性特征
* $w_1$、$w_2$、$w_3$、...、$w_n$为线性回归模型参数
* $b$为线性回归模型偏置项
* $h(x)$ 为模型预测输出



对$w$、$x$进行向量化表示:
$$
w = (w_1,w_2,w_3,...,w_n)^T \tag2
$$

$$
x = (x^{(1)},x^{(2)},x^{(3)},...,x^{(n)}) ^T \tag3
$$

将公式(2)、公式(3)，代入公式（1）得到：
$$
h(x) = w^Tx+b \tag4
$$


## 1.3 损失函数

线性回归的预测及真实标签均为实数值，损失函数可以使用MSE（Mean Squre Error，平均均方误差）。
$$
L = \frac{1}{2m}\sum_{i=1}^{m}[h(x_i)-y_i]^2 \tag5
$$
$$
L = \frac{1}{2m}\sum_{i=1}^{m}[w^Tx_i+b-y_i]^2 \tag6
$$
其中：

* m为样本数量
* $y_i$为第$i$ 个样本的真实标签
* 对MSE除以2是为了参数求解的方便

## 1.4 参数求解

根据1.3中提到的评估方法，通过变化模型参数可以降低误差使得模型达到最优。相反地，对误差进行极小化操作得到参数便是最优参数、最优模型，即：
$$
(w^{'}, b^{'}) = argmin_{(w,b)} \frac{1}{2m}\sum_{i=1}^{m}[w^Tx_i+b-y_i]^2 \tag7
$$
线性回归参数的常用求解方法包括：（1）解析解（最小二乘法）、（2）梯度下降法。

### 1.4.1 解析解（最小二乘法）

当偏导数为0时，函数取得极小值。
$$
\frac{\partial L}{\partial w} = \frac{\partial \frac{1}{2m}\sum_{i=1}^{m}[wx_i+b-y_i]^2}{\partial w} \tag8
$$
$$
\frac{\partial L}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(wx_i+b-y_i)x_i  \tag9
$$






### 1.4.2 梯度下降法







## 1.5 实现



## 1.6 总结






# 2. 逻辑回归（Logistic Regression）

## 2.1 模型

线性回归 $h(x) = w^Tx+b$ ，加入$Sigmoid$ 函数后值域为$[0,1]$，可以应用至分类分类问题，理解为逻辑回归。假设样本的标签为0和1， $h(x)$ 为取得预测为标签1的概率。
$$
h(x) = \sigma (w^Tx + b)\tag{11}
$$
其中， $ \sigma(x)$ 为 $Sigmoid$ 函数
$$
\sigma(x) = \frac{1}{1+e^{-x}} \tag{12}
$$
即：
$$
h(x) = \frac{1}{1+e^{-w^Tx+b}} \tag{13}
$$
先对假说模型进行简化，将偏置项$bias$看作为1并添加置特征$x$中。即：
$$
h(x) = \frac{1}{1+e^{-w^Tx}} \tag{14}
$$
根据定义，
$$
p(y=1|x) = h(x) = \frac{1}{1+e^{-w^Tx}} \tag{15}
$$

$$
p(y=0|x) = 1- h(x) = 1- \frac{1}{1+e^{-w^Tx}} = \frac{e^{-w^Tx}}{1+e^{-w^Tx}} \tag{16}
$$

## 2.2 损失函数

逻辑回归是一个分类问题，其极大似然函数可以设计如下：
$$
L(w) = \prod_{i=1}^N p(y=1|x_i)^{y_i}  p(y=0|x_i)^{1-y_i}  = \prod_{i=1}^N h(x_i)^{y_i}(1-h(x_i))^{1-y_i} \tag{17}
$$
对数极大似然函数如下：
$$
log(L(w)) = log( \prod_{i=1}^N p(y=1|x_i)^{y_i}  p(y=0|x_i)^{1-y_i} ) \tag{18}
$$

$$
log(L(w)) =  \sum_{i=1}^N  (y_i log(h(x_i))  + (1-y_i)log(1-h(x_i)) ) \tag{19}
$$

观察发现，对数极大似然函数与二分类任务的交叉熵损失函数相同，这是因为交叉熵损失函数是从概率分布KL散度推导出来，从根本上描述了预测概率分布与真实概率分布之间。而极大似然函数通过巧妙的设计得到了相似得结果。

## 2.3 参数求解

$$
\begin{align}
	\frac{ \partial J(w) }{\partial w} &= -\frac{1}{N}\sum_{i=1}^N \{ \frac{y_i}{h(x_i)} \frac{\partial h(x_i)}{\partial w} - \frac{1-y_i}{1-h(x_i)} \frac{\partial h(x_i)}{\partial w}  \} \\
	&= -\frac{1}{N}\sum_{i=1}^N \{ \frac{y_i}{h(x_i)} -  \frac{1-y_i}{1-h(x_i)} \}\frac{\partial h(x_i)}{\partial w}  \\
	&=  -\frac{1}{N}\sum_{i=1}^N \{ \frac{y_i}{h(x_i)} -  \frac{1-y_i}{1-h(x_i)} \} h(x_i)(1-h(x_i)) \frac{\partial w^Tx_i}{\partial w} \\
	&= -\frac{1}{N}\sum_{i=1}^N \{ y_i (1-h(x_i)) -(1-y_i)h(x_i) \} x_i \\
	&= -\frac{1}{N}\sum_{i=1}^N \{ y_i - h(x_i)\} x_i
\end{align}
$$

## 2.4 正则项









# 3. 背诵项

* 逻辑回归损失函数（二分类交叉熵损失函数）

$$
Loss =  \sum_{i=1}^N  (y_i log(h(x_i))  + (1-y_i)log(1-h(x_i)) ) \tag{19}
$$

* 逻辑回归梯度下降公式

$$
grad(w) =  -\frac{1}{N}\sum_{i=1}^N \{ y_i - h(x_i)\} x_i \tag{20}
$$





# 4.参考文献

https://zhuanlan.zhihu.com/p/74874291