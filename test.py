import time
import torch
import numpy as np
from envs.my_env import InvertedPendulumEnv
from envs.extended_env import ExtendedPendulumEnv, ExtendedObsEnv
from ppo.baseline_ppo import PPO2
from ppo.my_ppo import PPO

from argparse import ArgumentParser


def basic_test():
    env = InvertedPendulumEnv(max_reset_pos=0.01, n_iterations=1, reward_type=0)
    model_upswing = PPO(env)
    model_upswing.actor.load_state_dict(torch.load('snapshots/ppo_actor_cartpole_upswing.pth'))
    model_upswing.critic.load_state_dict(torch.load('snapshots/ppo_critic_cartpole_upswing.pth'))

    model_hold = PPO(env)
    model_hold.actor.load_state_dict(torch.load('snapshots/ppo_actor_cartpole_hold.pth'))
    model_hold.critic.load_state_dict(torch.load('snapshots/ppo_critic_cartpole_hold.pth'))

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


def extended_test():
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


def baseline_test():
    env = InvertedPendulumEnv(max_reset_pos=0.6, n_iterations=1600, reward_type=1)
    model = PPO2.load('snapshots/bound_expand_model', env=env)

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

def get_model_from_cli():
    parser = ArgumentParser('PPO')
    parser.add_argument('model',
                        type=str,
                        action='store',
                        help='configuration of model to test')
    return parser.parse_args()


def main(args=get_model_from_cli()):
    print(args)
    if args.model == 'baseline':
        baseline_test()
    elif args.model == 'basic':
        basic_test()
    elif args.model == 'extended_obs':
        extended_test()
    else:
        print('model not recognized')


main()