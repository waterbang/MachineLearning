# 线性回归

这里的递推过程跟动态规划很像，都是局部最优解推导全局最优解。

## 梯度下降

目标函数：$J(\theta) =$ ${1} \over {2m}$$\displaystyle\sum^{m}_{i=1}(y^i-h_\theta(x^i))^2$

### 批量梯度下降

容易得到最优解，但是速度慢，不适用大数据量。

$$
{\partial J(\theta) \over {\partial \theta_j} } = -{1 \over {2m}} \displaystyle\sum^{m}_{i=1}(y^i-h_\theta(x^i))x^i_j
$$

$y^i$标签，$h_\theta(x^i)$预测值，$x^i_j$数据样本本来存在的。

$$
\theta_j =  \theta_j + {1 \over {m}} \displaystyle\sum^{m}_{i=1}(y^i-h_\theta(x^i))x^i_j
$$


### 随机梯度下降

速度快，但随机。

$$
\theta_j =  \theta_j + (y^i-h_\theta(x^i))x^i_j
$$


### 小批量梯度下降法

上面两种方法的则中办法，实用，平常用这个。


$$
\theta_j:=  \theta_j - \alpha{1 \over {10}} \displaystyle\sum^{i+9}_{k=i}(y^k-h_\theta(x^k))x^k_j
$$

