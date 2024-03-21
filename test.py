import time
import mujoco
import mujoco.viewer
import numpy as np
from my_env import InvertedPendulumEnv
from my_ppo import PPO2
import mlflow

env = InvertedPendulumEnv(max_reset_angle=np.pi / 6, max_reset_pos=0.6, n_iterations=1600)
model = PPO2.load('bound_expand_model', env=env)



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
    action = model.policy.predict(ob)
    ob, reward, terminated, terminated, info = env.step(action[0])
    time.sleep(0.01)
