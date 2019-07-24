
import os
from time import sleep

import numpy as np
from tqdm import tqdm, trange

from maddpg.common import ReplayBuffer
from maddpg.algorithms import Maddpg, Coma


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_tqdm(*args):
    tqdm.write(' '.join(str(arg) for arg in args))


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
    print_tqdm(multi_env.observation_space.spaces)
    print_tqdm(multi_env.action_space.spaces)
    print_tqdm([space.shape for space in multi_env.action_space.spaces.values()])
    agents = Coma(multi_env.observation_space, multi_env.action_space,
                  shared_policy=False, shared_critic=False)  # ,
    # normalize={'observation': True, 'reward': True})
    print_tqdm('Starting...')
    exp_replay = ReplayBuffer(1e6)
    total_reward = None
    global_step = 0
    for _ in trange(400000):
        episode_reward = 0
        states_last = states = multi_env.reset()
        done = False
        step = 0
        while not done:
            actions = agents.predict(states)
            states, reward, done, info = multi_env.step(actions)
            episode_reward += reward
            step += 1
            if step > 25:
                if total_reward is None:
                    total_reward = episode_reward
                else:
                    total_reward = total_reward * 0.99 + episode_reward * 0.01
                done = True
            rewards = {key: reward for key in states}
            dones = {key: done for key in states}
            all_results = states_last, actions, rewards, states, dones
            exp_replay.add(*all_results)
            # all_results = {
            #    name: values for name, values in zip_map(*all_results)
            # }

            # for key, (replay, results) in zip_map(exp_replay, all_results):
            #    replay.add(*results)

            if global_step > 1024 and global_step % 100 == 0:
                stat, actio, rewar, stat_n, don = exp_replay.sample(1024)
                states_feed = stat
                actions_feed = actio
                rewards_feed = rewar
                states_n_feed = stat_n
                dones_feed = don
                # for key, replay in exp_replay.items():
                #    stat, actio, rewar, stat_n, don = replay.sample(1024)
                #    states_feed[key] = stat
                #    actions_feed[key] = actio
                #    rewards_feed[key] = rewar
                #    states_n_feed[key] = stat_n
                #    dones_feed[key] = don
                loss_before = agents.compute_loss(states_feed, actions_feed,
                                                  rewards_feed, states_n_feed,
                                                  dones_feed)
                agents.train_step(states_feed, actions_feed, rewards_feed,
                                  states_n_feed, dones_feed, global_step)
                loss_after = agents.compute_loss(states_feed, actions_feed,
                                                 rewards_feed, states_n_feed,
                                                 dones_feed)
                agents.update_targets()
            states_last = states
            global_step += 1

        if global_step > 1024 and global_step % 100 == 0:
            actor_loss_before = np.mean(list(loss_before['actor'].values()))
            critic_loss_before = np.mean(list(loss_before['critic'].values()))
            actor_loss_after = np.mean(list(loss_after['actor'].values()))
            critic_loss_after = np.mean(list(loss_after['critic'].values()))
            print_tqdm('*'*80)
            print_tqdm('Training:')
            print_tqdm('Stats:', info)
            print_tqdm('Total Reward:', total_reward)
            #print_tqdm('Grads Sum:', info['grads_sum'])
            #print_tqdm('Action Mean:', info['actions_mean'])
            #print_tqdm('Action Std:', info['actions_std'])
            #print_tqdm('Network Loss:',  info['loss'])
            #print_tqdm('Network Accu:', info['accuracy'])
            print_tqdm('Actor Loss Before:', actor_loss_before)
            print_tqdm('Actor Loss After: ', actor_loss_after)
            print_tqdm('Critic Loss Before:', critic_loss_before)
            print_tqdm('Critic Loss After: ', critic_loss_after)
            print_tqdm('*'*80)

        if global_step > 1024 and global_step % 1000 == 0:
            states_last = states = multi_env.reset()
            total_reward = 0
            done = False
            step = 0
            while not done:
                actions = agents.predict(states, False)
                states, reward, done, info = multi_env.step(actions)
                total_reward += reward
                step += 1
                states_last = states
                multi_env.render()
                sleep(0.1)
                if step > 25:
                    done = True
            print_tqdm('*'*80)
            print_tqdm('Evaluation:')
            print_tqdm('Total Reward:', total_reward)
            print_tqdm('*'*80)


if __name__ == '__main__':
    main()
