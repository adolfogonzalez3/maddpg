'''A module that contains the Maddpg class.'''
import shutil
import abc
from statistics import mean
from collections import namedtuple

import numpy as np
import numpy.random as npr
import tensorflow as tf
from tqdm import trange, tqdm

from maddpg.common import ReplayBuffer
from maddpg.common.utils_common import (map_to_batch, print_tqdm,
                                        get_shape, map_apply, create_default)


TrainInfo = namedtuple('TrainInfo', ['observations', 'rewards', 'dones',
                                     'infos', 'actor_loss', 'critic_loss',
                                     'step'])


class MultiAgentAlgBase(abc.ABC):
    '''
    A class for implementing mutiple agent TD3.
    '''

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.graph = tf.Graph()
        self._session = None
        self._saver = None
        with self.graph.as_default():
            self.placeholders = create_default(observation_space.spaces,
                                               action_space.spaces)

    @property
    def session(self):
        '''Get session and if it doesn't exist then create it.'''
        if self._session is None:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            with self.graph.as_default():
                init = tf.global_variables_initializer()
                self._saver = tf.train.Saver()
            self._session = tf.Session(graph=self.graph, config=config)
            self._session.run(init)
        return self._session

    def predict(self, observations, noisy=True):
        '''
        Predict next actions depending on observations.

        :param observations: ({array-like}) A map from agent names to arrays.
        :param noisy: (bool) If true then add noise otherwise no noise.
        :return: ({array-like}) Actions to take.
        '''
        observations = map_to_batch(observations,
                                    get_shape(self.observation_space))
        actions = self._predicts(observations)
        if noisy:
            def function(act):
                return np.squeeze(act + npr.normal(scale=0.2, size=act.shape))
        else:
            function = np.squeeze
        return map_apply(actions, function)

    def compute_values(self, observations):
        '''
        Predict value of taking the next predicted action.

        :param observations: ({array-like}) A map from agent names to arrays.
        :return: ({array-like}) Value of next predicted action.
        '''
        observations = map_to_batch(observations,
                                    get_shape(self.observation_space))
        return self._compute_values(observations)

    def compute_loss(self, observations, actions, rewards, observations_n,
                     dones):
        '''Compute the loss of the agents.'''
        observation_shapes = get_shape(self.observation_space)
        observations = map_to_batch(observations, observation_shapes)
        actions = map_to_batch(actions, get_shape(self.action_space))
        rewards = map_to_batch(rewards, {k: (1,) for k in rewards})
        observations_n = map_to_batch(observations_n, observation_shapes)
        dones = map_to_batch(dones, {k: (1,) for k in dones})
        return self._compute_loss(observations, actions, rewards,
                                  observations_n, dones)

    def train_step(self, observations, actions, rewards, observations_n,
                   dones, step=None):
        '''Train the agents.'''
        observation_shapes = get_shape(self.observation_space)
        observations = map_to_batch(observations, observation_shapes)
        actions = map_to_batch(actions, get_shape(self.action_space))
        rewards = map_to_batch(rewards, {k: (1,) for k in rewards})
        observations_n = map_to_batch(observations_n, observation_shapes)
        dones = map_to_batch(dones, {k: (1,) for k in dones})
        return self._train_step(observations, actions, rewards, observations_n,
                                dones, step=step)

    def learn_generator(self, env, timesteps=10**6, replay=None):
        '''
        Train the agents for some duration.

        :param time_steps: The number of time steps to train for.
        '''
        if replay is None:
            replay = ReplayBuffer(timesteps // 100)
        done = True
        for step in range(timesteps):
            if done:
                observations_last = env.reset()
            actions = self.predict(observations_last)
            observations, reward, done, infos = env.step(actions)
            rewards = {key: reward for key in observations}
            dones = {key: done for key in observations}
            replay.add(observations_last, actions, rewards,
                       observations, dones)
            observations_last = observations
            train_info = {}
            if step > 1024 and step % 5000 == 0:  # replay.can_sample():
                train_info = self.train_step(*replay.sample(1024), step)
                self.run_updates()
            yield TrainInfo(observations, rewards, dones, infos,
                            train_info.get('actor'),
                            train_info.get('critic'), step)

    def learn(self, env, timesteps=10**6, replay=None, verbose=True):
        '''
        Train the agents for some duration.

        :param time_steps: The number of time steps to train for.
        '''
        trainer = self.learn_generator(env, timesteps, replay)
        trainer = tqdm(trainer, total=timesteps) if verbose else trainer
        ep_reward = None
        total_reward = 0
        for training_info in trainer:
            done = any(training_info.dones.values())
            total_reward += mean(training_info.rewards.values())
            if done:
                if ep_reward:
                    ep_reward = ep_reward*.99 + total_reward*.01
                else:
                    ep_reward = total_reward
                total_reward = 0
            if training_info.actor_loss:
                if verbose:
                    actor_loss = mean(training_info.actor_loss.values())
                    critic_loss = mean(training_info.critic_loss.values())
                    columns, _ = shutil.get_terminal_size()
                    print_tqdm('*'*columns)
                    print_tqdm('Training Step:', training_info.step)
                    print_tqdm('Running Reward: {:+6.6f}'.format(ep_reward))
                    print_tqdm('Actor Loss:', actor_loss)
                    print_tqdm('Critic Loss:', critic_loss)
                    print_tqdm('*'*columns)

    def save(self, path):
        '''
        Save model to path.

        :param path: (str) A path to save the checkpoint to.
        '''
        self._saver.save(self._session, str(path))

    def load(self, path):
        '''
        Load model from path.

        :param path: (str) A path to load the checkpoint from.
        '''
        self._saver.restore(self._session, str(path))

    @abc.abstractmethod
    def run_updates(self):
        '''Run algorithm updates.'''

    @abc.abstractmethod
    def _predicts(self, observations):
        '''
        Predict next actions depending on observations.

        :param observations: ({array-like}) A map from agent names to arrays.
        :return: ({array-like}) Actions to take.
        '''

    @abc.abstractmethod
    def _compute_values(self, observations):
        '''
        Predict value of taking the next predicted action.

        :param observations: ({array-like}) A map from agent names to arrays.
        :return: ({array-like}) Value of next predicted action.
        '''

    @abc.abstractmethod
    def _train_step(self, observations, actions, rewards, observations_n,
                    dones, step=None):
        '''
        Train the agent.
        '''

    @abc.abstractmethod
    def _compute_loss(self, observations, actions, rewards, observations_n,
                      dones):
        '''
        Compute the loss of the algorithm.
        '''
