from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import time
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym


class InvertedPendulumEnv(gym.Env):
    xml_env = """
    <mujoco model="inverted pendulum">
            <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="160" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        </asset>
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="1" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.0 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-3 3"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
            <!--geom name="ground" type="plane" pos="0 0 0" /-->
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule" group="3"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-100000 100000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
        </actuator>
    </mujoco>
    """

    def __init__(
        self,  max_reset_pos, n_iterations, reward_type
    ):
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.model = mujoco.MjModel.from_xml_string(InvertedPendulumEnv.xml_env)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.action_space = gym.spaces.Box(-20.0, 20.0, (1,), np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        #self.num_envs = 1

        # data below is used for the extending boundaries experiment
        self.max_reset_pos = max_reset_pos
        self.n_iterations = n_iterations
        self.bound_angle = 0.01
        self.bound_pos = 0.01
        self.counter = 0
        self.pos_step = np.round(max_reset_pos / n_iterations, 6)

        #reward flag for configuring exps
        self.reward_type = reward_type

        # termination borders
        self.x_limit = 1.5
        self.theta_limit = np.pi / 90


        self.reset_model()

    def step(self, a):
        self.data.ctrl = a
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        done = False

        ob = self.obs()

        # two_pi = 2 * np.pi
        # reward_theta = (np.e ** (np.cos(ob[1]) + 1.0) - 1.0)
        # reward_x = np.cos((ob[0] / 5) * (np.pi / 2.0))
        # reward_theta_dot = (np.cos(ob[1]) * (np.e ** (np.cos(ob[3]) + 1.0) - 1.0) / two_pi) + 1.0
        # reward_x_dot = ((np.cos(ob[1]) * (np.e ** (np.cos(ob[2]) + 1.0) - 1) / two_pi) + 1.0)
        # reward = (reward_theta + reward_x + reward_theta_dot + reward_x_dot) / 4.0

        if np.cos(ob[1]) > 0:
            reward = np.cos(ob[1]) * 5 - np.abs(np.sin(ob[1])) * 3 + (np.abs(ob[0]) < 0.1) * 2
            if np.abs(ob[1]) < np.pi / 8 and np.abs(ob[3]) < 0.5:
                reward = 6 * np.cos(ob[1]) - 3 * np.abs(ob[0]) + (np.abs(ob[0]) < 0.05) * 2
        else:
            reward = 0

        # if np.abs(ob[1]) < self.theta_limit:
        #     done = True
        #     reward += 20
        # else:
        #     reward = np.abs(np.sin(ob[1])) + np.sqrt(np.abs(ob[2])) + np.sqrt(np.abs(ob[3])) - 2 * np.abs(ob[0])

        # if np.abs(ob[1]) < np.pi / 8 and np.abs(ob[3]) < 0.5:
        #     if self.reward_type == 0:
        #         reward = 5 - np.abs(ob[0])
        #     elif self.reward_type == 1:
        #         reward = 6 * (np.cos(ob[1])) - 3 * np.abs(ob[0]) + (np.abs(ob[0]) < 0.05) * 2
        # else:
        #     reward = 0

        terminated = bool((not np.isfinite(ob).all()))

        return ob, reward, terminated, done, {}


    def obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def reset(self, seed=None, options=None):
        obs = self.reset_model()
        return obs, {}

    def reset_model(self):

        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel
        #self.data.qpos[1] = 3.14  # Set the pole to be facing down
        return self.obs()

    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def draw_ball(self, position, color=[1, 0, 0, 1], radius=0.01):
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=np.array(position),
            mat=np.eye(3).flatten(),
            rgba=np.array(color),
        )
        self.viewer.user_scn.ngeom = 1

    @property
    def current_time(self):
        return self.data.time

    def close(self):
        pass