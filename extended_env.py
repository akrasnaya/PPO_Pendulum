import time
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym

SEED = 14
np.random.seed(SEED)

class ExtendedPendulumEnv(gym.Env):
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

    def __init__(self):
        # Initial positions
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)


        # Model data
        self.model = mujoco.MjModel.from_xml_string(self.xml_env)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.action_space = gym.spaces.Box(-3.0, 3.0, (1,), np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

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

        two_pi = 2 * np.pi
        reward_theta = (np.e ** (np.cos(ob[1]) + 1.0) - 1.0)
        reward_x = np.cos((ob[0] / 5) * (np.pi / 2.0))
        reward_theta_dot = (np.cos(ob[1]) * (np.e ** (np.cos(ob[3]) + 1.0) - 1.0) / two_pi) + 1.0
        reward_x_dot = ((np.cos(ob[1]) * (np.e ** (np.cos(ob[2]) + 1.0) - 1) / two_pi) + 1.0)
        reward = (reward_theta + reward_x + reward_theta_dot + reward_x_dot) / 4.0

        if np.cos(ob[1]) > 0:
            reward = np.cos(ob[1]) * 5 - np.abs(np.sin(ob[1])) * 3 + (np.abs(ob[0]) < 0.1) * 2
            if np.abs(ob[1]) < np.pi / 8 and np.abs(ob[3]) < 0.5:
                reward = 6 * np.cos(ob[1]) - 3 * np.abs(ob[0]) + (np.abs(ob[0]) < 0.05) * 2

        # if np.abs(ob[1]) < self.theta_limit:
        #     done = True
        #     reward += 20
        # else:
        #     reward = np.abs(np.sin(ob[1])) + np.sqrt(np.abs(ob[2])) + np.sqrt(np.abs(ob[3])) - 2 * np.abs(ob[0])


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
        self.data.qpos[1] = 3.14  # Set the pole to be facing down
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


class ExtendedObsEnv(gym.ObservationWrapper):
    def __init__(self, env, ball_generation):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.last_update = 0
        self.ball_update_time = ball_generation

    def observation(self, obs, ball_pos):
        obs = np.zeros(5)
        obs[:4] = np.concatenate([self.data.qpos, self.data.qvel]).ravel()
        obs[4] = ball_pos
        return obs

    def reset(self, seed=None, options=None):
        obs = self.reset_model()
        return obs, {}

    def reset_model(self):
        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel
        self.data.qpos[1] = 3.14  # Set the pole to be facing down
        return self.observation(self.obs(), 0)

    def step(self, a):
        self.data.ctrl = a
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        done = False

        target_pos = [0, 0, 0.6]
        if self.env.current_time - self.last_update > self.ball_update_time:
            target_pos = [np.random.rand() - 0.5, 0, 0.6]
            self.env.draw_ball(target_pos, radius=0.05)
            self.last_update = self.env.current_time

        ob = self.observation(self.obs(), target_pos[0])

        two_pi = 2 * np.pi
        reward_theta = (np.e ** (np.cos(ob[1]) + 1.0) - 1.0)
        reward_x = np.cos((ob[0] / 5) * (np.pi / 2.0)) + 3 * np.cos(ob[4] - ob[0])
        reward_theta_dot = (np.cos(ob[1]) * (np.e ** (np.cos(ob[3]) + 1.0) - 1.0) / two_pi) + 1.0
        reward_x_dot = ((np.cos(ob[1]) * (np.e ** (np.cos(ob[2]) + 1.0) - 1) / two_pi) + 1.0)
        reward = (reward_theta + reward_x + reward_theta_dot + reward_x_dot) / 4.0

        if np.cos(ob[1]) > 0:
            ball_pos = np.array([target_pos[0], target_pos[2]])
            pend_pos = np.array([0.6 * np.cos(ob[1]), 0.6 * np.cos(ob[1])])
            dist = np.linalg.norm(ball_pos - pend_pos)
            assert dist != 0
            reward = np.cos(ob[1]) * 5 - np.abs(np.sin(ob[1])) * 3 + (np.abs(ob[0]) < 0.1) * 2 + (1 / dist)
            if np.abs(ob[1]) < np.pi / 8 and np.abs(ob[3]) < 0.5:
                reward = 6 * np.cos(ob[1]) - 3 * np.abs(ob[0]) + (np.abs(ob[0]) < 0.05) * 2

        terminated = bool((not np.isfinite(ob).all()))

        return ob, reward, terminated, done, {}

