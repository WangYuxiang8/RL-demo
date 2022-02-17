"""
定义基于模型的agent子类
该类的主要功能是封装了环境模型、策略、以及真实环境这几个类实例
向外提供了一系列接口，使得实现变得简单
"""
from .base_agent import BaseAgent

class MBAgent(BaseAgent):

    def __init__(self):
        pass

    def train(self):
        pass

    def add_to_replay_buffer(self, paths):
        pass

    def sample(self, batch_size):
        pass