import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    safe_mean,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
from typing import TypeVar
import mlflow
from datetime import datetime
import matplotlib.pyplot as plt
import random

import torch as th
from torch.nn import functional as F

from my_env import InvertedPendulumEnv

# PPO2(policy="MlpPolicy", env=InvertedPendulumEnv()).learn(1000000, callback=None)

SelfPPO = TypeVar("SelfPPO", bound="PPO")

SEED = 14
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


class PPO2(PPO):
    # def __init__(self):
    #     super().__init__()
    #     mlflow.set_tracking_uri("http://127.0.0.1:5000")

    def plot_obs(self, action_tensor, obs_tensor, epoch):
        # plot and log obs
        figure, axis = plt.subplots(5, 1, figsize=(30, 20))

        axis[0].plot(obs_tensor[:, 0], color="r")
        axis[0].set_title("Cart position")

        axis[1].plot(obs_tensor[:, 1], color="g")
        axis[1].set_title("Pendulum angle")

        axis[2].plot(obs_tensor[:, 2])
        axis[2].set_title("Cart velocity")

        axis[3].plot(obs_tensor[:, 3], color="c")
        axis[3].set_title("Pendulum angular velocity")

        axis[4].plot(action_tensor, color="m")
        axis[4].set_title("Actions")

        if epoch%500 == 0:
            plt.savefig(f"obs/obs_epoch_{str(epoch).rjust(5, '0')}.png")
            plt.close()

            mlflow.log_artifact(
                f"obs/obs_epoch_{str(epoch).rjust(5, '0')}.png",
                "obs",
            )
        else:
            plt.close()

    def train(self, iter):
        # mlflow.set_tracking_uri("http://127.0.0.1:5000")

        """Update policy using the currently gathered rollout buffer."""

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        obs_tensor = torch.Tensor([[0, 0, 0, 0]])
        act_tensor = torch.Tensor([[0]])

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            actor_loss_per_epoch = []
            critic_loss_per_epoch = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                act_tensor = torch.cat((act_tensor, actions), dim=0)

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )


                obs_tensor = torch.cat((obs_tensor, rollout_data.observations), dim=0)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # mlflow.log_metrics({"actor_loss": policy_loss}, step=epoch)

                # Logging
                pg_losses.append(policy_loss.item())
                actor_loss_per_epoch.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                # mlflow.log_metrics({"critic_loss": value_loss}, step=epoch)
                value_losses.append(value_loss.item())
                critic_loss_per_epoch.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            # mlflow.log_metrics(
            #     {
            #         f"critic_loss in iteration {str(iter).rjust(5, '0')}": np.mean(
            #             critic_loss_per_epoch
            #         )
            #     },
            #     step=epoch,
            # )
            # mlflow.log_metrics(
            #     {
            #         f"actor_loss in iteration {str(iter).rjust(5, '0')}": np.mean(
            #             actor_loss_per_epoch
            #         )
            #     },
            #     step=epoch,
            # )

            self._n_updates += 1
            if not continue_training:
                break

        self.plot_obs(act_tensor, obs_tensor, iter)
        # mlflow.log_metrics({f"critic_loss in iteration {iter}": np.mean(value_losses)}, step=epoch)
        # mlflow.log_metrics({f"actor_loss in iteration {iter}": np.mean(pg_losses)}, step=epoch)
        mlflow.log_metric("Total Reward", self.rollout_buffer.rewards.sum(), step=iter)

        #self.env.set_boundaries(iter)
        self.env.reset()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):

        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        experiment_name = f"ppo cartpole hold"
        now = datetime.now()
        #run_name = f"steps_{self.n_steps}_" + now.strftime("%m/%d/%Y, %H:%M:%S")
        run_name = f"BOUND_EXPAND_LEARN_{self.n_steps}_STEPS_" + now.strftime("%m/%d/%Y, %H:%M:%S")
        mlflow.set_experiment(experiment_name)

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None


        with mlflow.start_run(run_name=run_name):
            for py_file in Path(__file__).parent.glob("*.py"):
                mlflow.log_artifact(str(py_file), f"code")

            while self.num_timesteps < total_timesteps:
                continue_training = self.collect_rollouts(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                )

                if not continue_training:
                    break

                iteration += 1
                self._update_current_progress_remaining(
                    self.num_timesteps, total_timesteps
                )

                self.train(iteration)


            # Saving actor network
            torch.save(self.policy.action_net.state_dict(), f'models/ppo_actor_cartpole_iter{str(iteration)}.pth')
            mlflow.log_artifact(f'models/ppo_actor_cartpole_iter{str(iteration)}.pth',
                                'models')

            # Saving critic network
            torch.save(self.policy.value_net.state_dict(), f'models/ppo_critic_cartpole_iter{str(iteration)}.pth')
            mlflow.log_artifact(f'models/ppo_critic_cartpole_iter{str(iteration)}.pth',
                                'models')

            torch.save(self.policy.state_dict(), f'models/ppo_policy_cartpole_iter{str(iteration)}.pth')
            mlflow.log_artifact(f'models/ppo_policy_cartpole_iter{str(iteration)}.pth',
                                'models')

            #self.save(f'model_{str(iteration)}')

        callback.on_training_end()

        return self