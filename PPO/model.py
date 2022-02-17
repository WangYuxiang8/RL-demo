"""
策略网络，pytorch实现
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std=0.6):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, action_dim),
            nn.Tanh()
        )
        self.action_var = torch.full((action_dim,), action_std * action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        # 多元高斯分布
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action, action_logprob

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        # 扩展维度，因为输入state可能是batch型的
        action_var = self.action_var.expand_as(action_mean)
        # 多元高斯分布，diag_embed对每个维度的张量都会对角化
        action_cov = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, action_cov)
        action_logprob = dist.log_prob(action)
        # 动作概率分布的熵，表明分布的是否均匀，与探索度有关
        dist_entropy = dist.entropy()
        return action_logprob, dist_entropy


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self):
        raise NotImplementedError

    def evaluate(self, state):
        return self.critic(state)
