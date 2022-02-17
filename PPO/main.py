"""
PPO算法和DPPO算法实现
"""
import gym
import torch
from agent import PPO

# 超参数
ENV_NAME = "LunarLanderContinuous-v2"  # Gym环境名称
MAX_TIMESTEPS = int(3e5)  # 总训练回合数
MAX_EP_LEN = 1000  # 单回合最大训练次数
UPDATE_INTERVAL = MAX_EP_LEN * 4  # 更新网络的步长
PRINT_INTERVAL = MAX_EP_LEN * 10  # 打印一次信息的时间
GAMMA = 0.99  # 奖赏衰减
LR = 0.001  # 学习率
E = 0.2  # clip函数范围


def set_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    return device


def main():
    """
    主循环
    """
    env = gym.make(ENV_NAME)
    agent = PPO(env.observation_space.shape[0], env.action_space.shape[0], GAMMA, LR, E, device=set_device())

    time_step = 0
    while time_step <= MAX_TIMESTEPS:
        observation = env.reset()
        reward_sum = 0
        for _ in range(MAX_EP_LEN):
            # env.render()
            time_step += 1
            action, action_logprob = agent.select_action(observation)
            observation_, reward, done, _ = env.step(action)
            agent.store_transition(observation, action, reward, action_logprob, done)
            # 训练agent
            if time_step % UPDATE_INTERVAL == 0:
                agent.update(10, 15)
            if time_step % PRINT_INTERVAL == 0:
                print("Timestep: {}, Reward: {}".format(time_step, reward_sum))
            if done:
                break
            observation = observation_
            reward_sum += reward


if __name__ == '__main__':
    main()
