from ppo import networks

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np
import mlflow
from datetime import datetime
from pathlib import Path
import random

SEED = 14
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

class PPO:
    def __init__(self, env, **hyperparameters):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self._init_hyperparameters(hyperparameters)

        # Initialize actor and critic networks
        self.actor = networks.Actor(self.obs_dim, self.act_dim)
        self.critic = networks.Critic(self.obs_dim, 1)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize variables for covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 1024
        self.max_timesteps_per_episode = 400
        self.gamma = 0.99
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.00025
        self.ent_coef = 1.37976e-07
        self.grad_norm = 0.5
        self.gae_lam = 0.9 # GAE Lambda

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

    def evaluate(self, batch_obs, batch_acts):
        """
        :param batch_obs:
        :return: value V for each obs in batch_obs
        """
        V = self.critic(batch_obs).squeeze()
        mean_act = self.actor(batch_obs)
        dist = MultivariateNormal(mean_act, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy() # in order to add entropy regularization

    def compute_rtgs(self, batch_rews):
        """
        :param rews:
        :return: the rewards-to-go per episode per batch
        """
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in ep_rews:
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        # Quering the actor network for a mean action
        mean_act = self.actor(obs)
        dist = MultivariateNormal(mean_act, self.cov_mat)

        # Sampling an action and getting it's log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_gae(self, rewards, values, dones):
        batch_advs = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_adv = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.gae_lam * (1 - ep_dones[t]) * last_adv
                last_adv = advantage
                advantages.insert(0, advantage)

            batch_advs.extend(advantages)

        #batch_advs = np.array(batch_advs)

        return torch.tensor(batch_advs, dtype=torch.float)

    def rollout(self):
        """

        :return: rollout data
        """
        # Batch data
        batch_obs = []          # batch observations
        batch_acts = []         # batch actions
        batch_log_probs = []    # log probs of each action
        batch_rews = []         # batch rewards
        batch_rtgs = []         # batch rewards-to-go
        batch_lens = []         # episodic lenghts in batch
        batch_vals = []
        batch_dones = []


        t = 0                   # Number of timesteps run so far in this batch

        while t < self.timesteps_per_batch:
            ep_rews = []        # Episode rewards
            ep_vals = []
            ep_dones = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                ep_dones.append(done)
                t += 1

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                val = self.critic(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_rews.append(rew)
                ep_vals.append(val)

                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic lenghts and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        print(len(batch_rews), len(batch_rews[0]))



        # Convert to np.array
        batch_obs = np.array(batch_obs)
        batch_acts = np.array(batch_acts)
        batch_rews = np.array(batch_rews)
        batch_rtgs = np.array(batch_rtgs)
        batch_log_probs = np.array(batch_log_probs)
        batch_lens = np.array(batch_lens)
        batch_dones = np.array(batch_dones)
        #batch_vals = np.array(batch_vals)


        # Reshape data as tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rews = torch.tensor(batch_rews, dtype=torch.float)
        #batch_vals = torch.tensor(batch_vals, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens, batch_vals, batch_dones

    def learn(self, total_timesteps):
        # Set logging
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        experiment_name = f"ppo cartpole hold"
        now = datetime.now()
        run_name = "UPSWING_" + now.strftime("%m/%d/%Y, %H:%M:%S")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            for py_file in Path(__file__).parent.glob("*.py"):
                mlflow.log_artifact(str(py_file), f"../code")

            t_so_far = 0    # Timesteps simulated so far
            while t_so_far < total_timesteps:
                batch_obs, batch_acts, batch_log_probs, batch_rews, \
                    batch_rtgs, batch_lens, batch_vals, batch_dones = self.rollout()

                # Adding GAE calculation
                A_k = self.compute_gae(batch_rews, batch_vals, batch_dones)
                #V, _, _ = self.evaluate(batch_obs, batch_acts)
                V = self.critic(batch_obs).squeeze()
                batch_rtgs = A_k + V.detach()

                # Calculate advantage
                #A_k = batch_rtgs - V.detach()

                # Normalizing advantage
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                t_so_far += np.sum(batch_lens)

                for epoch in range(self.n_updates_per_iteration):
                    # Learning rate annealing
                    new_lr = max(0.0, self.lr * (1.0 - ((t_so_far - 1.0) / total_timesteps)))
                    self.actor_optimizer.param_groups[0]['lr'] = new_lr
                    self.critic_optimizer.param_groups[0]['lr'] = new_lr
                    mlflow.log_metric('Learning rate', new_lr, step=epoch)

                    # Calculate pi_theta(a_t | s_t)
                    V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

                    # Approximating KL Divergence
                    logratios = curr_log_probs - batch_log_probs
                    ratios = torch.exp(logratios)
                    kl_div = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate loss
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    # Adding entropy regularization
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    mlflow.log_metric('Actor loss', actor_loss, step=epoch)

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                    self.actor_optimizer.step()

                    critic_loss = nn.MSELoss()(V, batch_rtgs)

                    mlflow.log_metric('Critic loss', critic_loss, step=epoch)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                    self.critic_optimizer.step()

                mlflow.log_metric("Total Reward", batch_rews.sum(), step=t_so_far)

            # Saving actor network
            torch.save(self.actor.state_dict(),
                       f'../models/ppo_actor_cartpole_hold11.pth')
            mlflow.log_artifact(f'../models/ppo_actor_cartpole_hold11.pth',
                                'models')

            # Saving critic network
            torch.save(self.critic.state_dict(),
                       f'../models/ppo_critic_cartpole_hold11.pth')
            mlflow.log_artifact(f'../models/ppo_critic_cartpole_hold11.pth',
                                    'models')

