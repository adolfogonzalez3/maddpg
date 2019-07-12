
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm, trange

from maddpg.common.utils_common import zip_map
from maddpg.algorithms.maddpg import Maddpg
from maddpg.trainer.replay_buffer import ReplayBuffer
from custom_envs.envs import MultiOptLRs


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_tqdm(*args):
    tqdm.write(' '.join(str(arg) for arg in args))


def main(batch_size=1):
    multi_env = MultiOptLRs(data_set='mnist', max_batches=100,
                            batch_size=128, max_history=25)
    agents = Maddpg(multi_env.observation_space,
                    multi_env.action_space,
                    shared_policy=True, shared_critic=True)
    print_tqdm('Starting...')
    exp_replay = {name: ReplayBuffer(1e6)
                  for name in multi_env.action_space.spaces}
    global_step = 0
    last_info = defaultdict(lambda: None)
    for _ in trange(60000):
        total_reward = 0
        states_last = states = multi_env.reset()
        done = False
        all_actions = []
        while not done:
            actions = agents.predict(states)
            actions = {key: np.squeeze(act) for key, act in actions.items()}
            states, reward, done, info = multi_env.step(actions)
            
            if done:
                last_info = info
            #print(np.any(np.isnan(list(actions.values()))))
            total_reward += reward
            rewards = {key: reward for key in states}
            dones = {key: done for key in states}
            all_results = states_last, actions, rewards, states, dones
            all_results = {
                name: values for name, values in zip_map(*all_results)
            }

            for key, (replay, results) in zip_map(exp_replay, all_results):
                replay.add(*results)

            if global_step > batch_size and global_step % 100 == 0:
                states_feed = {}
                actions_feed = {}
                rewards_feed = {}
                states_n_feed = {}
                dones_feed = {}
                idxs = {}
                for key, replay in exp_replay.items():
                    #idx, mem, _ = replay.sample(1024)
                    #idxs[key] = idx
                    #stat, actio, rewar, stat_n, don = mem
                    stat, actio, rewar, stat_n, don = replay.sample(batch_size)
                    states_feed[key] = stat
                    actions_feed[key] = actio
                    rewards_feed[key] = rewar
                    states_n_feed[key] = stat_n
                    dones_feed[key] = don
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
                all_actions = [list(act.values()) for act in all_actions]
                print_tqdm('*'*80)
                print_tqdm('Training:')
                print_tqdm('Total Reward:', total_reward)
                print_tqdm('Stats:', last_info['episode'])
                print_tqdm('Grads Sum:', last_info['grads_sum'])
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
            all_actions.append(actions)
    agents.save('optimizer/model.ckpt')


if __name__ == '__main__':
    main()
