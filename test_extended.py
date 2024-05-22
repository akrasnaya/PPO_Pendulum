import time
import torch
import mujoco
import mujoco.viewer
import numpy as np
from my_env import InvertedPendulumEnv
from extended_env import ExtendedPendulumEnv, ExtendedObsEnv
from ppo import PPO
import mlflow

env = ExtendedPendulumEnv()
env.set_dt(0.02)
env = ExtendedObsEnv(env, ball_generation=1)

model = PPO(env)
model.actor.load_state_dict(torch.load('snapshots/ppo_actor_cartpole_extended.pth'))
model.critic.load_state_dict(torch.load('snapshots/ppo_critic_cartpole_extended.pth'))


ob, dict = env.reset()

target_pos = [0, 0, 0.6]
while env.current_time < 5000:
    ob = np.array(ob)
    action = model.get_action(ob)
    ob, reward, terminated, terminated, info = env.step(action[0])
    time.sleep(0.01)
