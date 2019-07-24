
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from maddpg.algorithms import Maddpg
from maddpg.common.replaybuffer import ReplayBuffer
from custom_envs.envs import MultiOptLRs
from custom_envs.utils.utils_logging import Monitor


def print_tqdm(*args):
    tqdm.write(' '.join(str(arg) for arg in args))


def run(path, episodes=600, batch_size=10):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / 'training_log.csv'
    multi_env = MultiOptLRs(data_set='mnist', max_batches=100,
                            batch_size=32, max_history=25, reward_version=0,
                            observation_version=3, action_version=0,
                            version=5)
    multi_env = Monitor(multi_env, log_path,
                        info_keywords=('loss', 'accuracy', 'actions_mean',
                                       'weights_mean', 'actions_std',
                                       'states_mean', 'grads_mean'),
                        chunk_size=5)
    agents = Maddpg(multi_env.observation_space, multi_env.action_space,
                    shared_policy=True, shared_critic=True)
    print_tqdm('Starting...')
    global_step = 0
    pretrain_max = 1e3
    exp_replay = ReplayBuffer(1e4)
    last_info = defaultdict(lambda: None)
    pretraining = True
    total_reward = None
    for _ in trange(episodes):
        states_last = states = multi_env.reset()
        done = False
        episode_reward = 0
        while not done:
            if pretraining:
                actions = multi_env.action_space.sample()
            else:
                actions = agents.predict(states)
                actions = {key: np.squeeze(act)
                           for key, act in actions.items()}
            states, reward, done, info = multi_env.step(actions)
            episode_reward += reward
            if done:
                last_info = info
                if total_reward:
                    total_reward = total_reward * 0.99 + episode_reward * 0.01
                else:
                    total_reward = episode_reward
            rewards = {key: reward for key in states}
            dones = {key: done for key in states}
            all_results = states_last, actions, rewards, states, dones
            exp_replay.add(*all_results)

            if global_step > 16 and global_step % 100 == 0:
                pretraining = False
                stat, actio, rewar, stat_n, don = exp_replay.sample(1)
                states_feed = stat
                actions_feed = actio
                rewards_feed = rewar
                states_n_feed = stat_n
                dones_feed = don
                loss_before = agents.compute_loss(states_feed, actions_feed,
                                                  rewards_feed, states_n_feed,
                                                  dones_feed)
                losses = agents.train_step(states_feed, actions_feed,
                                           rewards_feed, states_n_feed,
                                           dones_feed)
                losses = agents.compute_loss(states_feed, actions_feed,
                                             rewards_feed, states_n_feed,
                                             dones_feed)
                agents.update_targets()

                actor_loss_before = np.mean(
                    list(loss_before['actor'].values()))
                critic_loss_before = np.mean(
                    list(loss_before['critic'].values()))
                actor_loss = np.mean(list(losses['actor'].values()))
                critic_loss = np.mean(list(losses['critic'].values()))
                print_tqdm('*'*80)
                print_tqdm('Step:', global_step)
                print_tqdm('Training:')
                print_tqdm('Total Reward:', total_reward)
                print_tqdm('Stats:', last_info['episode'])
                print_tqdm('Grads Sum:', last_info['grads_sum'])
                print_tqdm('States Mean:', last_info['states_mean'])
                print_tqdm('Action Mean:', last_info['actions_mean'])
                print_tqdm('Action Std:', last_info['actions_std'])
                print_tqdm('Network Loss:',  last_info['loss'])
                print_tqdm('Network Accu:', last_info['accuracy'])
                print_tqdm('Actor Loss Before:', actor_loss_before)
                print_tqdm('Critic Loss Before:', critic_loss_before)
                print_tqdm('Actor Loss:', actor_loss)
                print_tqdm('Critic Loss:', critic_loss)
                print_tqdm('*'*80)

            states_last = states
            global_step += 1
    agents.save(path / 'model.ckpt')


def main():
    '''Evaluate a trained model against logistic regression.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", help="The path to save the results.",
                        type=Path)
    parser.add_argument("--episodes", help="The number of episodes.",
                        type=int, default=600)
    parser.add_argument("--data_set", help="The data set to trial against.",
                        type=str, default='iris')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    run(args.save_path)


if __name__ == '__main__':
    main()
