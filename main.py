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
    parser.add_argument('model',
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


def main(args=get_model_from_cli()):
    print(args)
    if args.model == 'baseline':
        PPO2(env=InvertedPendulumEnv(max_reset_pos=0.01,
                                     n_iterations=1,
                                     reward_type=0),
             device="cpu", **kwargs).learn(5120000.0, callback=None)
    elif args.model == 'transfer':
        model = PPO2.load('snapshots/baseline_model.zip', env=InvertedPendulumEnv(max_reset_pos=0.01,
                                                                        n_iterations=1,
                                                                        reward_type=1))
        model.learn(614400.0, callback=None)
    elif args.model == 'bound_ext':
        model = PPO2.load('snapshots/baseline_trans_model.zip', env=InvertedPendulumEnv(max_reset_pos=0.6,
                                                                              n_iterations=1600,
                                                                              reward_type=1))
        model.learn(3072000.0, callback=None)
    else:
        print('model not recognized')


main()