from my_env import InvertedPendulumEnv
from networks import Actor, Critic

from torch.distributions import MultivariateNormal
from torch.optim import Adam
import mlflow

class PPO:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = InvertedPendulumEnv.observation_space.shape[0]
        self.act_dim = InvertedPendulumEnv.action_space.shape[0]
        self._init_hyperparameters()

        # Initialize actor and critic networks
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim, 1)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize variables for covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def evaluate(self, batch_obs, batch_acts):
        """
        :param batch_obs:
        :return: value V for each obs in batch_obs
        """
        V = self.critic(batch_obs).squeeze()
        mean_act = self.actor(batch_obs)
        dist = MultivariateNormal(mean_act, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

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

        return action.detatch().numpy(), log_prob.detatch()

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

        t = 0                   # Number of timesteps run so far in this batch

        while t < self.timesteps_per_batch:
            ep_rews = []        # Episode rewards

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic lenghts and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def learn(self):
        # Set logging
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        experiment_name = f"ppo cartpole hold"
        now = datetime.now()
        run_name = f"BOUND_EXPAND_LEARN_{self.n_steps}_STEPS_" + now.strftime("%m/%d/%Y, %H:%M:%S")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            for py_file in Path(__file__).parent.glob("*.py"):
                mlflow.log_artifact(str(py_file), f"code")

            t_so_far = 0    # Timesteps simulated so far
            while t_so_far < total_timesteps:
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

                t_so_far += np.sum(batch_lens)

                V, _ = self.evaluate(batch_obs)

                # Calculate advantage
                A_k = batch_rtgs - V.detatch()

                # Normalizing advantage
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                for epoch in self.n_updates_per_iteration:
                    # Calculate pi_theta(a_t | s_t)
                    V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                    ratios = torch.exp(curr_log_probs - batch_log_probs)

                    # Calculate surrogate loss
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optimizer.step()

                    critic_loss = nn.MSELoss()(V, batch_rtgs)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    mlflow.log_metric("Total Reward", batch_rtgs.sum(), step=epoch)

            # Saving actor network
            torch.save(self.actor.state_dict(),
                        f'models/ppo_actor_cartpole_iter{str(iteration)}.pth')
            mlflow.log_artifact(f'models/ppo_actor_cartpole_iter{str(iteration)}.pth',
                                'models')

            # Saving critic network
            torch.save(self.critic.state_dict(),
                        f'models/ppo_critic_cartpole_iter{str(iteration)}.pth')
            mlflow.log_artifact(f'models/ppo_critic_cartpole_iter{str(iteration)}.pth',
                                    'models')

