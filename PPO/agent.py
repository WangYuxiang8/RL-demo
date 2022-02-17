"""
PPO强化学习代理
"""
import torch

from storage import ReplayBuffer
from model import Actor, Critic


class PPO(object):
    def __init__(self, state_dim, action_dim, gamma, lr, e, device):
        self.actor = Actor(state_dim, action_dim)  # 策略网络
        self.critic = Critic(state_dim)  # 价值网络
        self.buffer = ReplayBuffer()  # 记忆库
        self.gamma = gamma  # 奖赏衰减
        self.lr = lr  # 学习率
        self.e = e  # clip范围
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.mse_loss = torch.nn.MSELoss()
        self.device = device

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储一个字段信息到replay buffer
        """
        self.buffer.store(state, action, reward, next_state, done)

    def select_action(self, state):
        """
        选动作
        Args:
            state 输入当前状态
        Return:
            action 输出当前应该执行的动作
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, action_logprob = self.actor.act(state)
        return action.tolist(), action_logprob.item()

    def update(self, actor_eps, critic_eps):
        """
        计算梯度并更新参数
        Args:
            actor_eps 策略网络的更新回合
            critic_eps 价值网络的更新回合
        """
        # 计算Advantage
        rewards = []
        cumulate = 0
        for r, d in zip(self.buffer.reward, self.buffer.done):
            if d:
                cumulate = 0
            cumulate = self.gamma * cumulate + r
            rewards.insert(0, cumulate)

        old_states = torch.tensor(self.buffer.state)
        old_actions = torch.tensor(self.buffer.action)
        old_logprobs = torch.tensor(self.buffer.logprobs)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 更新actor参数
        for _ in range(actor_eps):
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            eval_values = self.critic.evaluate(old_states)
            advantages = rewards - torch.squeeze(eval_values)
            r_theta = torch.exp(logprobs - old_logprobs)
            loss = torch.min(r_theta * advantages, torch.clamp(r_theta, 1 - self.e, 1 + self.e) * advantages)
            # 梯度更新
            self.actor_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()

        # 更新critic参数
        for _ in range(critic_eps):
            eval_values = self.critic.evaluate(old_states)
            loss = self.mse_loss(torch.squeeze(eval_values), rewards)
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.critic_optimizer.step()

        self.buffer.clear()
