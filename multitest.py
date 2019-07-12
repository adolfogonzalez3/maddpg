
from maddpg.common.utils_common import zip_map
from custom_envs.envs import MultiOptLRs
from maddpg.trainer.replay_buffer import ReplayBuffer
from maddpg.trainer.prioritized_replay_buffer import PrioritizedReplayMemory
from maddpg.algorithms.maddpg import Maddpg
import numpy as np
import os
from time import sleep
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)
    return env


def main():
    multi_env = make_env("simple_spread")
    # multi_env = MultiOptLRs(data_set='mnist', max_batches=40)
    print(multi_env.observation_space.spaces)
    print(multi_env.action_space.spaces)
    agents = Maddpg(multi_env.observation_space.spaces,
                    multi_env.action_space.spaces,
                    shared_policy=True, shared_critic=True)
    print('Starting...')
    exp_replay = {name: ReplayBuffer(1e6)
                  for name in multi_env.action_space.spaces}
    global_step = 0
    for _ in range(60000):
        total_reward = 0
        states_last = states = multi_env.reset()
        done = False
        step = 0
        while not done:
            actions = agents.predict(states)
            states, reward, done, info = multi_env.step(actions)
            total_reward += reward
            step += 1
            if step > 25:
                done = True
            rewards = {key: reward for key in states}
            dones = {key: done for key in states}
            all_results = states_last, actions, rewards, states, dones
            all_results = {
                name: values for name, values in zip_map(*all_results)
            }

            for key, (replay, results) in zip_map(exp_replay, all_results):
                replay.add(*results)

            if global_step > 1024 and global_step % 100 == 0:
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
                    stat, actio, rewar, stat_n, don = replay.sample(1024)
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
                #for _, (replay, idx, loss) in zip_map(exp_replay, idxs,
                #                                      losses['critic']):
                #    replay.batch_update(idx, np.sqrt(loss))
                agents.update_targets()
            states_last = states
            global_step += 1

        if global_step > 1024 and global_step % 100 == 0:
            actor_loss_before = np.mean(list(loss_before['actor'].values()))
            critic_loss_before = np.mean(list(loss_before['critic'].values()))
            actor_loss = np.mean(list(losses['actor'].values()))
            critic_loss = np.mean(list(losses['critic'].values()))
            print('*'*80)
            print('Training:')
            print('Stats:', info)
            print('Total Reward:', total_reward)
            #print('Grads Sum:', info['grads_sum'])
            #print('Action Mean:', info['actions_mean'])
            #print('Action Std:', info['actions_std'])
            #print('Network Loss:',  info['loss'])
            #print('Network Accu:', info['accuracy'])
            print('Actor Loss Before:', actor_loss_before)
            print('Critic Loss Before:', critic_loss_before)
            print('Actor Loss:', actor_loss)
            print('Critic Loss:', critic_loss)
            print('*'*80)

        if global_step > 1024 and global_step % 1000 == 0:
            states_last = states = multi_env.reset()
            total_reward = 0
            done = False
            step = 0
            while not done:
                actions = agents.predict(states)
                print(actions)
                states, reward, done, info = multi_env.step(actions)
                total_reward += reward
                step += 1
                states_last = states
                multi_env.render()
                sleep(0.1)
                if step > 25:
                    done = True
            print('*'*80)
            print('Evaluation:')
            print('Total Reward:', total_reward)
            print('*'*80)


if __name__ == '__main__':
    main()
