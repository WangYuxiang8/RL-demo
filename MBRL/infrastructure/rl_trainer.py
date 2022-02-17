"""
强化学习的训练模型
"""
from MBRL.infrastructure import utils

class RLTrainer(object):

    def __init__(self, params):
        self.params = params

    def run_training_loop(self, n_iter, collect_policy):
        """
        训练主循环
        Args:
            n_iter 总共迭代回合
            collect_policy 采集轨迹数据时采用的策略（Random shooting, CEM）
        """

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)

            # 本轮需要采集的轨迹数据数量
            # 最开始要采集多一点
            use_batch_size = self.params['batch_size']
            if itr == 0:
                use_batch_size = self.params['batch_size_initial']

            # 采集轨迹数据
            paths = self.collect_training_trajectories(collect_policy, use_batch_size)

            # 将本次采集到的数据加入到记忆库中
            self.agent.add_to_replay_buffer(paths, self.params['add_sl_noise'])

            # 训练一次模型（这里的模型指的是模拟环境的模型）
            print("\nTraining agent...")
            self.train_agent()

    def collect_training_trajectories(self, collect_policy, num_transitions_to_sample):
        """
        收集训练需要的轨迹数据，agent与环境交互产生一系列的轨迹数据，并保存到记忆库中
        Args:
            collect_policy 采集轨迹数据时采用的策略（Random shooting, CEM）
            num_transitions_to_sample 和batch_size一致 - 每轮训练时，收集轨迹数据的迭代步数
        Returns:
            paths 多条轨迹，每条轨迹包含ep_len长的数据，表示agent与环境的交互数据
                其中每个元素由 (s_t+1, s_t, a_t, r_t, done) 组成
        """

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, num_transitions_to_sample,
                                                               self.params['ep_len'])

        return paths

    def train_agent(self):
        """
        调用agent接口来训练模型model(这个模型指的是环境模型)
        从记忆库中采样train_batch_size个样本来进行训练
        记录训练日志并返回
        """

        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'])

            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

