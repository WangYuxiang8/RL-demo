"""
记忆库，用来存储交互数据
"""


class ReplayBuffer(object):
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.logprobs = []
        self.done = []

    def store(self, state, action, reward, logprobs, done):
        """
        存储一个字段信息
        """
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.logprobs.append(logprobs)
        self.done.append(done)

    def sample(self):
        """
        采样数据以供训练
        """

    def clear(self):
        """
        清除所有数据
        """
        self.__init__()

