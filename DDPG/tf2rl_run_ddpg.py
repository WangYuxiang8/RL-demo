import gym
from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer


parser = Trainer.get_argument()
parser = DDPG.get_argument(parser)
args = parser.parse_args()

env = gym.make("HalfCheetah-v2")
test_env = gym.make("HalfCheetah-v2")
policy = DDPG(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=0,  # Run on CPU. If you want to run on GPU, specify GPU number
    memory_capacity=10000,
    max_action=env.action_space.high[0],
    batch_size=32,
    n_warmup=500)
trainer = Trainer(policy, env, args, test_env=test_env)
trainer()