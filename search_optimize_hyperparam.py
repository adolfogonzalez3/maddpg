'''Run an multi agent experiment.'''
import os
import argparse
from pathlib import Path
from functools import partial

import gym
import optuna
import numpy as np
import tensorflow as tf
from tqdm import trange

from maddpg.algorithms import Maddpg
from maddpg.common.replaybuffer import ReplayBuffer
from custom_envs.utils.utils_logging import Monitor
import custom_envs.utils.utils_file as utils_file


def run_agents(parameters, trial):
    path = Path(parameters['path'])
    log_path = path / 'training_log.csv'
    multi_env = Monitor(
        gym.make(parameters['env_name'], **parameters['kwargs']), log_path,
        info_keywords=(
            'loss', 'accuracy', 'actions_mean',
            'weights_mean', 'actions_std',
            'states_mean', 'grads_mean'
        ),
        chunk_size=parameters.setdefault('chunk_size', 5)
    )
    agents = Maddpg(multi_env.observation_space, multi_env.action_space,
                    shared_policy=True, shared_critic=True,
                    hyperparameters=parameters)
    current_episode = 0
    exp_replay = ReplayBuffer(parameters['replay_buffer_size'])
    total_reward = None
    done = True
    with trange(parameters['total_timesteps'], leave=False) as timesteps:
        for global_step in timesteps:
            pretraining = global_step < parameters['replay_buffer_size']
            if done:
                states_last = states = multi_env.reset()
                episode_reward = 0

            if pretraining:
                actions = multi_env.action_space.sample()
            else:
                actions = agents.predict(states)
                actions = {key: np.squeeze(act)
                           for key, act in actions.items()}
            states, reward, done, info = multi_env.step(actions)
            episode_reward += reward
            if done:
                current_episode += 1
                trial.report(info['loss'], current_episode)
                if trial.should_prune():
                    raise optuna.structs.TrialPruned()
                if total_reward:
                    total_reward = total_reward * 0.99 + episode_reward * 0.01
                else:
                    total_reward = episode_reward
            rewards = {key: reward for key in states}
            dones = {key: done for key in states}
            all_results = states_last, actions, rewards, states, dones
            exp_replay.add(*all_results)
            if not pretraining and global_step % 100 == 0:
                sample = exp_replay.sample(1)
                agents.train_step(*sample)
                agents.update_targets()
                desc = 'Timestep: {:8d} Avg Reward: {:+4.4f}'
                desc = desc.format(global_step, total_reward)
                timesteps.set_description(desc)
            states_last = states
    agents.save(path / 'model.ckpt')
    multi_env.close()
    return info['loss']


def run_experiment(parameters, trial):
    '''Set up and run an experiment.'''
    parameters = parameters.copy()
    batch_size = [2**i for i in range(5, 12)]
    batch_size = int(trial.suggest_categorical('batch_size', batch_size))
    obs_version = int(trial.suggest_int('observation_version', 0, 3))
    max_history = int(trial.suggest_discrete_uniform('max_history', 5, 50, 5))
    learning_rate = float(trial.suggest_loguniform('learning_rate', 1e-5, 1e0))
    buffer_size = int(trial.suggest_int('replay_buffer_size', 1e3, 1e5))
    parameters.update({
        'gamma': float(trial.suggest_uniform('gamma', 0.1, 1.0)),
        'replay_buffer_size': buffer_size
    })
    parameters.setdefault('agent_parameters', {}).update({
        'policy': {
            'learning_rate': learning_rate
        },
        'critic': {
            'learning_rate': learning_rate
        }
    })
    parameters.setdefault('kwargs', {}).update({
        'batch_size': batch_size,
        'version': int(trial.suggest_int('version', 0, 5)),
        'observation_version': obs_version, 'max_history': max_history,
        'reward_version': int(trial.suggest_int('reward_version', 0, 1)),
        'action_version': int(trial.suggest_int('action_version', 0, 1))
    })
    path = Path(parameters['path']) / str(trial.number)
    path.mkdir()
    parameters['path'] = str(path)
    parameters['commit'] = utils_file.get_commit_hash(Path(__file__).parent)
    utils_file.save_json(parameters, path / 'parameters.json')
    return run_agents(parameters, trial)


def main():
    '''Main script function.'''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to save the results.")
    parser.add_argument("--alg", help="The algorithm to use.",
                        default='MADDPG', type=str.upper)
    parser.add_argument("--env_name", help="The gamma to use.",
                        default='MultiOptLRs-v0')
    parser.add_argument('--total_timesteps', default=int(1e6), type=int,
                        help="Number of timesteps per training session")
    parser.add_argument('--data_set', help="The data set to use.",
                        default='mnist')
    parser.add_argument('--trials', help="The number of trials to run.",
                        default=10, type=int)
    args = parser.parse_args()
    parameters = vars(args).copy()
    del parameters['trials']
    path = Path(parameters['path'])
    if not path.exists():
        path.mkdir()
        parameters['kwargs'] = {'data_set': parameters['data_set']}
        utils_file.save_json(parameters, path / 'parameters.json')
    else:
        if (path / 'study.db').exists():
            print('Directory exists. Using existing study and parameters.')
            parameters = utils_file.load_json(path / 'parameters.json')
        else:
            raise FileExistsError(('Directory already exists and is not a '
                                   'study.'))
    objective = partial(run_experiment, parameters)
    storage = 'sqlite:///' + str(path / 'study.db')
    study = optuna.create_study(study_name=str(path.name),
                                storage=storage, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)


if __name__ == '__main__':
    main()
