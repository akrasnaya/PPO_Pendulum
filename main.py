import time
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
from typing import TypeVar
import mlflow

import torch
from torch.nn import functional as F

from my_env import InvertedPendulumEnv
from ppo import PPO
from typing import OrderedDict
from argparse import ArgumentParser


# def get_model_from_cli():
#     parser = ArgumentParser('PPO')
#     parser.add_argument('model',
#                         type=str,
#                         action='store',
#                         help='configuration of model to learn')
#     return parser.parse_args()
#
# SelfPPO = TypeVar("SelfPPO", bound="PPO")
# kwargs = OrderedDict([
#         ("batch_size", 64),
#         ("clip_range", 0.4),
#         ("ent_coef", 1.37976e-07),
#         ("gae_lambda", 0.9),
#         ("gamma", 0.999),
#         ("learning_rate", 0.000222425),
#         ("max_grad_norm", 0.3),
#         ("n_epochs", 5),
#         ("n_steps", 1024),
#         ("policy", "MlpPolicy"),
#         ("vf_coef", 0.19816),
#         # ("normalize_kwargs", {"norm_obs": True, "norm_reward": False}),
#     ])
#
#
# def main(args=get_model_from_cli()):
#     print(args)
#     if args.model == 'baseline':
#         PPO2(env=InvertedPendulumEnv(max_reset_pos=0.01,
#                                      n_iterations=1,
#                                      reward_type=0),
#              device="cpu", **kwargs).learn(5120000.0, callback=None)
#     elif args.model == 'transfer':
#         model = PPO2.load('snapshots/baseline_model.zip', env=InvertedPendulumEnv(max_reset_pos=0.01,
#                                                                         n_iterations=1,
#                                                                         reward_type=1))
#         model.learn(614400.0, callback=None)
#     elif args.model == 'bound_ext':
#         model = PPO2.load('snapshots/baseline_trans_model.zip', env=InvertedPendulumEnv(max_reset_pos=0.6,
#                                                                               n_iterations=1600,
#                                                                               reward_type=1))
#         model.learn(3072000.0, callback=None)
#     else:
#         print('model not recognized')

def main():
    hyperparameters = {
        'timesteps_per_batch': 1024,
        'max_timesteps_per_episode': 100,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
    }
    env = InvertedPendulumEnv(max_reset_pos=0.01, n_iterations=1, reward_type=0)
    env.set_dt(0.05)
    model = PPO(env)
    # model.actor.load_state_dict(torch.load('models/ppo_actor_cartpole_hold3.pth'))
    # model.critic.load_state_dict(torch.load('models/ppo_critic_cartpole_hold3.pth'))
    model.learn(2560000)

main()