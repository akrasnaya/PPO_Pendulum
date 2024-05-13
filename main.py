from my_env import InvertedPendulumEnv
from ppo import PPO
from argparse import ArgumentParser




def main():
    hyperparameters = {
        'timesteps_per_batch': 1024,
        'max_timesteps_per_episode': 400,
        'gamma': 0.99,
        'n_updates_per_iteration': 5,
        'lr': 0.00025,
        'clip': 0.2,
        'ent_coef': 1.37976e-07,
        'grad_norm': 0.5,
        'gae_lam': 0.9
    }

    env = InvertedPendulumEnv(max_reset_pos=0.01, n_iterations=1, reward_type=0)
    env.set_dt(0.05)
    model = PPO(env, **hyperparameters)
    model.learn(2560000)

main()