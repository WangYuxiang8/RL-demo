"""
工具库，包括以下工具函数：
    - 采样轨迹数据
"""
import numpy as np
import time


def Path(obs, acs, rewards, next_obs, terminals):
    """
        轨迹数据结构，记录一条轨迹数据，用分离数组的方式记录各个字段
        包含 (s_t+1, s_t, a_t, r_t, done) 数据
    """

    return {"observation": np.array(obs, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectory(env, policy, max_path_length, render=False):
    """
    采样一条轨迹数据，和sample_trajectories参数差不多
    """

    # 每次开始都要重置环境
    ob = env.reset()

    # 初始化需要记录的值
    # 当前观测、动作、奖励、下一时刻观测、终止状态
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0

    while True:

        # 渲染模拟环境
        if render:
            env.render()
            time.sleep(env.model.opt.timestep)

        # 根据当前的观测值来决定采取什么动作
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        # 向环境执行动作，获得反馈
        ob, rew, done, _ = env.step(ac)

        # 记录
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # rollout 可能因为任务结束或者到达了horizon而终止
        rollout_done = int(done or steps == max_path_length)
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """
    采样多条轨迹数据

    Args:
        env 交互环境
        policy 采集轨迹数据时采用的策略（Random shooting, CEM）
        min_timesteps_per_batch 和batch_size一致 - 每轮训练时，收集轨迹数据的迭代步数
        max_path_length 每次收集轨迹数据时轨迹的horizon长度，和ep_len一致
        render 是否边渲染环境边训练
    Returns:
        paths 多条轨迹，每条轨迹包含ep_len长的数据，表示agent与环境的交互数据
                其中每个元素由 (s_t+1, s_t, a_t, r_t, done) 组成
    """

    # 记录做了多少次交互
    # 这个值按理说应该等于min_timesteps_per_batch
    # 但有的任务可能提前done了，有的任务达到了max_path_length，所以加起来的和不一定是那个值
    timesteps_this_batch = 0

    # 所有的rollouts
    paths = []

    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += get_path_length(path)

        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')
    return paths, timesteps_this_batch


def get_path_length(path):
    """
    返回输入path(一次rollout的交互过程记录)的长度，即本次rollout走了多长
    这里在path中随便取一个值的len都可以
    """

    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    """
    对模型的输入进行标准化
    """
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    """
    解标准化，从标准化数据转到非标准化数据
    """
    return data * std + mean
