'''A module that contains the Maddpg class.'''

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from maddpg.modules import MaddpgModule
import maddpg.common.utils_common as utils
from maddpg.common.utils_common import convert_spaces_to_placeholders
from maddpg.common.utils_common import zip_map


class Maddpg:
    '''
    A class for implementing MADDPG.
    '''

    def __init__(self, observation_space, action_space, shared_policy=False,
                 shared_critic=False):
        self.observation_space = observation_space
        self.action_space = action_space
        observation_spaces = observation_space.spaces
        action_spaces = action_space.spaces
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        graph = tf.Graph()
        with graph.as_default():
            observations = convert_spaces_to_placeholders(observation_spaces)
            actions = convert_spaces_to_placeholders(action_spaces)
            rewards = {name: tf.placeholder(tf.float32, shape=(None, 1))
                       for name in observations}
            observations_n = convert_spaces_to_placeholders(observation_spaces)
            dones = {name: tf.placeholder(tf.float32, shape=(None, 1))
                     for name in observations}
            print('Starting...')
            maddpg_module = MaddpgModule(observation_spaces, action_spaces,
                                         shared_policy, shared_critic)
            functions = maddpg_module(observations, actions, rewards,
                                      observations_n, dones)
            self._saver = tf.train.Saver()
            print('Initializing...')
            init = tf.global_variables_initializer()
        self._session = tf.Session(graph=graph, config=config)
        self._session.run(init)
        tf.summary.FileWriter('summaries', self._session.graph)
        self._predicts = utils.TfFunction(
            self._session, inputs=observations,
            outputs=functions.actor_predict
        )
        self._compute_values = utils.TfFunction(
            self._session, inputs=observations,
            outputs=functions.critic_predict
        )
        updates = [functions.actor_update, functions.critic_update]
        print('Updates:', updates)
        self._update_targets = utils.TfFunction(self._session, updates=updates)
        optimizers = [functions.actor_optimizer, functions.critic_optimizer]
        print('Opts:', optimizers)
        inputs = utils.flatten_map({'obs': observations, 'act': actions,
                                    'rew': rewards, 'obs_next': observations_n,
                                    'done': dones})
        outputs = utils.flatten_map({'actor': functions.actor_losses,
                                     'critic': functions.critic_losses})
        self._train = utils.TfFunction(self._session, inputs=inputs,
                                       updates=optimizers, outputs=outputs)
        self._compute_loss = utils.TfFunction(self._session, inputs=inputs,
                                              outputs=outputs)
        self.functions = functions

    def predict(self, observations):
        '''Predict next action based on observation.'''
        observations = {name: np.reshape(obs, (-1,) + space.shape)
                        for name, (obs, space) in
                        zip_map(observations, self.observation_space.spaces)}
        return self._predicts(observations)

    def update_targets(self):
        '''Update the target networks for all agents.'''
        self._update_targets()

    def train_step(self, observations, actions, rewards, observations_n,
                   dones):
        '''Train the agents.'''
        observations = {name: np.reshape(obs, (-1,) + space.shape)
                        for name, (obs, space) in
                        zip_map(observations, self.observation_space.spaces)}
        actions = {name: np.reshape(act, (-1,) + space.shape)
                   for name, (act, space) in
                   zip_map(actions, self.action_space.spaces)}
        rewards = {name: np.reshape(reward, (-1, 1))
                   for name, reward in rewards.items()}
        observations_n = {
            name: np.reshape(obs, (-1,) + space.shape)
            for name, (obs, space) in zip_map(observations_n,
                                              self.observation_space.spaces)
        }
        dones = {name: np.reshape(done, (-1, 1))
                 for name, done in dones.items()}
        feed_dict = utils.flatten_map({'obs': observations, 'act': actions,
                                       'rew': rewards, 'done': dones,
                                       'obs_next': observations_n})
        return utils.unflatten_map(self._train(feed_dict))

    def compute_loss(self, observations, actions, rewards, observations_n,
                     dones):
        '''Compute the loss of the agents.'''
        observations = {name: np.reshape(obs, (-1,) + space.shape)
                        for name, (obs, space) in
                        zip_map(observations, self.observation_space.spaces)}
        actions = {name: np.reshape(act, (-1,) + space.shape)
                   for name, (act, space) in
                   zip_map(actions, self.action_space.spaces)}
        rewards = {name: np.reshape(reward, (-1, 1))
                   for name, reward in rewards.items()}
        observations_n = {
            name: np.reshape(obs, (-1,) + space.shape)
            for name, (obs, space) in zip_map(observations_n,
                                              self.observation_space.spaces)
        }
        dones = {name: np.reshape(done, (-1, 1))
                 for name, done in dones.items()}
        feed_dict = utils.flatten_map({'obs': observations, 'act': actions,
                                       'rew': rewards, 'done': dones,
                                       'obs_next': observations_n})
        return utils.unflatten_map(self._compute_loss(feed_dict))

    def save(self, path):
        '''
        Save model to path.

        :param path: (str) A path to save the checkpoint to.
        '''
        self._saver.save(self._session, path)

    def load(self, path):
        '''
        Load model from path.

        :param path: (str) A path to load the checkpoint from.
        '''
        self._saver.restore(self._session, path)
