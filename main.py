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

import torch as th
from torch.nn import functional as F

from my_env import InvertedPendulumEnv
from my_ppo import PPO2
from typing import OrderedDict

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
        ("n_steps", 32),
        ("policy", "MlpPolicy"),
        ("vf_coef", 0.19816),
        # ("normalize_kwargs", {"norm_obs": True, "norm_reward": False}),
    ])


PPO2(env=InvertedPendulumEnv(), device="cpu", **kwargs).learn(10000.0, callback=None)