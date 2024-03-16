from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, safe_mean
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

SelfPPO = TypeVar("SelfPPO", bound="PPO")


PPO2(policy="MlpPolicy", env=InvertedPendulumEnv()).learn(1000000, callback=None)