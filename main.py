from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    safe_mean,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

import time
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    safe_mean,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

import time
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
from typing import TypeVar
import mlflow

import torch as th
from torch.nn import functional as F

from my_env import InvertedPendulumEnv
from my_ppo import PPO2
from typing import OrderedDict
from argparse import ArgumentParser


def get_model_from_cli():
    parser = ArgumentParser('PPO')
    parser.add_argument('-m',
                        'model',
                        type=str,
                        action='store',
                        help='configuration of model to learn')
    return parser.parse_args()

SelfPPO = TypeVar("SelfPPO", bound="PPO")
kwargs = OrderedDict([
        ("batch_size", 64),
        ("clip_range", 0.4),
        ("ent_coef", 1.37976e-07),
        ("gae_lambda", 0.9),
        ("gamma", 0.999),
        ("learning_rate", 0.000222425),
        ("max_grad_norm", 0.3),
        ("n_epochs", 5),
        ("n_steps", 1024),
        ("policy", "MlpPolicy"),
        ("vf_coef", 0.19816),
        # ("normalize_kwargs", {"norm_obs": True, "norm_reward": False}),
    ])


#model = PPO2(env=InvertedPendulumEnv(), device="cpu", **kwargs)
model = PPO2.load('baseline_trans_model.zip', env=InvertedPendulumEnv(max_reset_angle=np.pi / 6,
                                                                max_reset_pos=0.6,
                                                                n_iterations=1600))
#model.env = InvertedPendulumEnv()
#model.policy = th.load('models/ppo_policy_cartpole_iter5.pth')
#model.policy.action_net = th.load('models/ppo_actor_cartpole_iter5.pth')
#model.policy.value_net = th.load('models/ppo_critic_cartpole_iter5.pth')
model.learn(3072000.0, callback=None)
model.save('bound_expand_model_2')

#PPO2(env=InvertedPendulumEnv(), device="cpu", **kwargs).learn(5120000.0, callback=None)