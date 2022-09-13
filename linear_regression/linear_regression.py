"""Linear Regression Module"""
## 代码来自：https://github.com/trekhleb/homemade-machine-learning/blob/master/homemade/linear_regression/linear_regression.py
# Import dependencies.
import numpy as np

from ..utils.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        # 线性回归的构造函数。

        # :参数data:训练集。
        # :参数标签:训练集输出(正确值)。
        # :param多项式al_degree:附加多项式特征的程度。
        # :param sinusoid_degree:正弦特征乘法器。
        # :param normalize_data:表示特性应该规范化的标志。
        # 

        #规范化功能，添加功能栏。
        (
            data_processed,
            features_mean,
            features_deviation
        ) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 初始化模型参数
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, lambda_param=0, num_iterations=500):
        """线性回归。

          :学习率(梯度下降的步长)
          :param lambda_param:正则化参数
          :param num_iterations:梯度下降迭代次数。
        """

        # 梯度下降法。
        cost_history = self.gradient_descent(alpha, lambda_param, num_iterations)

        return self.theta, cost_history

    def gradient_descent(self, alpha, lambda_param, num_iterations):
        """梯度下降法。

        它计算每个参数应该采取的步骤(增量)
        为了使代价函数最小化。

        :学习率(梯度下降的步长)
        :param lambda_param:正则化参数
        :param num_iterations:梯度下降迭代次数。
        """

        cost_history = []

        for _ in range(num_iterations):
            # 在参数向量上执行一个梯度步骤。
            self.gradient_step(alpha, lambda_param)

            # 在每次迭代中保存成本
            cost_history.append(self.cost_function(self.data, self.labels, lambda_param))

        return cost_history

    def gradient_step(self, alpha, lambda_param):
        """渐变步骤。
         函数对 theta 参数执行一步梯度下降。
        :param alpha：学习率（梯度下降的步长）
        :param lambda_param: 正则化参数
        """

        # 计算训练样本的数量。
        num_examples = self.data.shape[0]

        # 对所有 m 个示例的假设预测。
        predictions = LinearRegression.hypothesis(self.data, self.theta)

        # 所有 m 个示例的预测值和实际值之间的差异。
        delta = predictions - self.labels

        # 计算正则化参数。
        reg_param = 1 - alpha * lambda_param / num_examples

        # 创建 theta 快捷方式。
        theta = self.theta

        #  梯度下降的向量化版本。
        theta = theta * reg_param - alpha * (1 / num_examples) * (delta.T @ self.data).T
        #  我们不应该正则化参数 theta_zero。
        theta[0] = theta[0] - alpha * (1 / num_examples) * (self.data[:, 0].T @ delta).T

        self.theta = theta

    def get_cost(self, data, labels, lambda_param):
        """获取特定数据集的成本值。
        :param data：训练或测试数据集。
        :param 标签：训练集输出（正确值）。
        :param lambda_param: 正则化参数
        """

        data_processed = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

        return self.cost_function(data_processed, labels, lambda_param)

    def cost_function(self, data, labels, lambda_param):
        """成本函数。
        它显示了我们的模型基于当前模型参数的准确度。
        :param data：训练或测试数据集。
        :param 标签：训练集输出（正确值）。
        :param lambda_param: 正则化参数
        """

        # 计算训练样本和特征的数量。
        num_examples = data.shape[0]

        # 获取预测值和正确输出值之间的差异。
        delta = LinearRegression.hypothesis(data, self.theta) - labels

        # 计算正则化参数。
        # 请记住，我们不应该正则化参数 theta_zero。
        theta_cut = self.theta[1:, 0]
        reg_param = lambda_param * (theta_cut.T @ theta_cut)

        # 计算当前预测成本。
        cost = (1 / 2 * num_examples) * (delta.T @ delta + reg_param)

        #  让我们从唯一的成本 numpy 矩阵单元中提取成本值。
        return cost[0][0]

    def predict(self, data):
        """根据训练的 theta 值预测 data_set 输入的输出
        :param data：训练特征集。
        """

        # 规范化特征并添加列。
        data_processed = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

        # 使用模型假设进行预测。
        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions

    @staticmethod
    def hypothesis(data, theta):
        """假设函数。
        它根据输入值 X 和模型参数预测输出值 y。
        :param data：用于计算预测的数据集。
        :param theta: 模型参数。
        :return: 模型根据提供的 theta 做出的预测。
        """

        predictions = data @ theta

        return predictions
