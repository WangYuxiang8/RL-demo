"""
定义环境模型，从 (s_t, a_t) -> (s_t+1 - s_t) 的映射
"""

import tensorflow as tf
from MBRL.model.base_model import BaseModel
from MBRL.infrastructure.utils import normalize, unnormalize
import numpy as np


class FFModel(tf.keras.Model, BaseModel):
    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        """
        定义模型层及相关变量
        Args:
            ac_dim: 动作空间维度 int
            ob_dim: 状态空间维度 int
            n_layers: 隐藏层数量 int
            size: 隐藏层维度（节点个数） int
            learning_rate: 学习率 float
        """
        super().__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate

        # 构建多层感知机神经网络模型
        self.input_layer = tf.keras.layers.Dense(self.ac_dim + self.ob_dim, activation=tf.nn.tanh)
        self.hidden_layers = []
        for _ in range(self.n_layers - 1):
            self.hidden_layers.append(tf.keras.layers.Dense(self.size, activation=tf.nn.tanh))
        self.output_layer = tf.keras.layers.Dense(self.ob_dim, activation=tf.keras.activations.linear)  # 激活函数不做改变

        # 定义优化器
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        # 定义模型输入和输出变量的均值和标准差，用于标准化
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def call(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        """
        定义模型前馈过程
        """
        # 正则化输入数据
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)

        # 连接状态和动作，形成输入维度
        concatenated_input = np.concatenate((obs_normalized, acs_normalized), axis=1)

        # 定义前馈过程
        x = self.input_layer(concatenated_input)
        for hidden in self.hidden_layers:
            x = hidden(x)
        delta_pred_normalized = self.output_layer(x)

        # 输出的是(s_t+1 - s_t)，所以要加上输入的状态s_t，才能得到下一个状态值
        next_obs_pred = obs_unnormalized + unnormalize(delta_pred_normalized, delta_mean, delta_std)

        return next_obs_pred, delta_pred_normalized

    def update(self, observations, actions, next_observations, data_statistics):
        """
        定义模型损失和梯度下降更新
        """
        # 计算正则化的目标值
        target = normalize(next_observations - observations, data_statistics['delta_mean'],
                           data_statistics['delta_std'])

        # 定义训练过程
        with tf.GradientTape() as tape:
            # 计算模型预测值
            _, pred_delta_normalized = self(observations, actions, **data_statistics)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=target, y_pred=pred_delta_normalized)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))

        return {
            "Training Loss": loss
        }

    def get_prediction(self, ob_no, ac_na, data_statistics):
        """
        定义模型预测输出
        """
        pred, _ = self(ob_no, ac_na, **data_statistics)
        return pred
