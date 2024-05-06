import time
import torch
import mujoco
import mujoco.viewer
import numpy as np
from my_env import InvertedPendulumEnv
from ppo import PPO
import mlflow

env = InvertedPendulumEnv(max_reset_pos=0.01, n_iterations=1, reward_type=0)
model = PPO(env)
model.actor.load_state_dict(torch.load('models/ppo_actor_cartpole_hold3.pth'))
model.critic.load_state_dict(torch.load('models/ppo_critic_cartpole_hold3.pth'))



last_update = 0
ob, dict = env.reset()

target_pos = [0, 0, 0.6]
while env.current_time < 5000:
    if env.current_time - last_update > 10:
        target_pos = [np.random.rand() - 0.5, 0, 0.6]
        env.draw_ball(target_pos, radius=0.05)
        last_update = env.current_time
    ob = np.array(ob)
    #ob[0] = np.clip((ob[0] - target_pos[0]), -0.4, 0.2)
    action, _ = model.get_action(ob)
    ob, reward, terminated, terminated, info = env.step(action[0])
    time.sleep(0.01)
