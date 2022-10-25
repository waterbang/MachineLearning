# Machine Learning notes


1. [linear regression](./linear_regression/README.md)

## 防止过拟合

1. 减少特征数
2. 增加数据量
3. 正则化

### 正则化（Regularized,规则化）

使用一些规则，达到削减解的空间，增加求出正确值的可能性。
### 范数（norm）

范数（norm）的概念来源于泛函分析，wiki中的定义相当简单明了：范数是具有“长度”概念的函数，用于衡量一个矢量的大小（测量矢量的测度）

    0范数，向量中非零元素的个数。
    1范数，为绝对值之和。
    2范数，就是通常意义上的模。
#### 正则化代价函数

L2正则化：
$$
{J(\theta)} = {{1} \over {2m}}\left[ \displaystyle\sum^{m}_{i=1}(h_\theta(x^i)-y^i)^2 + \lambda\displaystyle\sum^{n}_{j=1}\theta_j^2 \right]
$$

L1正则化

$$
{J(\theta)} = {{1} \over {2m}}\left[ \displaystyle\sum^{m}_{i=1}(h_\theta(x^i)-y^i)^2 + \lambda\displaystyle\sum^{n}_{j=1}｜\theta_j｜ \right]
$$

## 岭回归(Ridge Regression)

岭回归最早是用来处理特征数多于样本的情况，现在也用于在估计中加入偏差，从而得到更好的估计。
同时也可以解决多重共线性的问题。岭回归是一种有偏估计。

### 岭回归代价函数

$$
{J(\theta)} = {{1} \over {2m}}\left[ \displaystyle\sum^{m}_{i=1}(h_\theta(x^i)-y^i)^2 + \lambda\displaystyle\sum^{n}_{j=1}\theta_j^2 \right]
$$

#### 求解

$\lambda$为岭系数,I为单位矩阵(对角线上全为1,其他元素全为0)

$$
W = (X^TX+\lambda I)^{-1} X^Ty 
$$

## 弹性网(Elastic Net)

$$
{J(\theta)} = {{1} \over {2m}}\left[ \displaystyle\sum^{m}_{i=1}(h_\theta(x^i)-y^i)^2 + \lambda\displaystyle\sum^{n}_{j=1}｜\theta_j｜^q \right]
$$

![p-范数](http://markdown-img.waterbang.top/machineLearning20221025234559.png)