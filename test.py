import time
import torch
import mujoco
import mujoco.viewer
import numpy as np
from my_env import InvertedPendulumEnv
from ppo import PPO
import mlflow

env = InvertedPendulumEnv(max_reset_pos=0.01, n_iterations=1, reward_type=0)
model_upswing = PPO(env)
model_upswing.actor.load_state_dict(torch.load('models/ppo_actor_cartpole_hold9.pth'))
model_upswing.critic.load_state_dict(torch.load('models/ppo_critic_cartpole_hold9.pth'))

model_hold = PPO(env)
model_hold.actor.load_state_dict(torch.load('models/ppo_actor_cartpole_hold3.pth'))
model_hold.critic.load_state_dict(torch.load('models/ppo_critic_cartpole_hold3.pth'))



last_update = 0
ob, dict = env.reset()

target_pos = [0, 0, 0.6]
while env.current_time < 5000:
    if env.current_time - last_update > 10:
        target_pos = [np.random.rand() - 0.5, 0, 0.6]
        env.draw_ball(target_pos, radius=0.05)
        last_update = env.current_time
    ob = np.array(ob)
    ob[0] = np.clip((ob[0] - target_pos[0]), -0.4, 0.2)
    action_hold, _ = model_hold.get_action(ob)
    action_upswing, _ = model_upswing.get_action(ob)
    if np.cos(ob[1]) > 0:
        action = np.abs(np.cos(ob[1])) * action_hold + (1 - np.abs(np.cos(ob[1]))) * action_upswing
    else:
        action = action_upswing
    ob, reward, terminated, terminated, info = env.step(action[0])
    time.sleep(0.01)
