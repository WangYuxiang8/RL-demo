"""
OpenAI gym 简单环境测试
Cartpole-v0 环境
"""

import gym


def train_simple(env):
    env.reset()  # 重置环境
    for _ in range(1000):
        env.render()  # 渲染环境
        env.step(env.action_space.sample())  # 选择一个随机动作来执行，step就是让环境执行一步动作
    env.close()


def train_standard(env):
    for i_episode in range(200):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


def check_spaces(env):
    """
    检查动作空间和状态空间的维度和范围
    """
    print("Action space: {}".format(env.action_space))
    print("Observation space: {}, high: {}, low: {}".format(env.observation_space,
                                                            env.observation_space.high,
                                                            env.observation_space.low))

def check_all_env(env):
    """
    检查当前gym所有可用的环境
    """
    print(env.registry.all())


if __name__ == '__main__':
    """
    Environment options:
        CartPole-v0
        MountainCar-v0
        MsPacman-v0 - Atari
        Hopper-v2 - MuJoCo
    """
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--env', type=str, default='Carnival-ram-v0', help='choose one environment of gym')
    args = parse.parse_args()

    env = gym.make(args.env)
    # check_spaces(envs)
    # check_all_env(gym.envs)
    train_standard(env)
