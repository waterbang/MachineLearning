# 线性回归

<img src="http://markdown-img.waterbang.top/machineLearning/costFunction.jpg" height="200" width="500">

## 代价函数（Cost Function/loss Function）

真实值y,预测值$h_\theta(x)$,则误差平方和为$\displaystyle\sum^{m}_{i=1}(y^i-h_\theta(x^i))^2$

$$
{J(\theta_0,\theta_1)} = {{1} \over {2m}} \displaystyle\sum^{m}_{i=1}(y^i-h_\theta(x^i))^2
$$

这里的$1\over2$没有实际意义，只是为了后面求导方便约掉，不影响最终结果。

## 梯度下降

这里的递推过程跟动态规划很像，都是局部最优解推导全局最优解。

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

