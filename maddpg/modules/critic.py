'''A module that contains the Critic class.'''

from collections import namedtuple

import sonnet as snt
import tensorflow as tf

from maddpg.modules.laggingnetwork import LaggingNetwork

CriticReturn = namedtuple('PolicyReturn', ['predict', 'predict_target',
                                           'update_target'])


class Critic(LaggingNetwork):
    '''
    A class for implementing a critic.
    '''

    def __init__(self, observation_space, action_space, name='critic'):
        '''
        Create a critic.

        :param observation_space: (gym.space) A space compatible with gym.
        :param action_space: (gym.space) A space compatible with gym.
        :param name: (str) The name of the critic.
        '''
        super().__init__((64, 64, 1), name=name)
        self.observation_space = observation_space
        self.action_space = action_space

    @snt.reuse_variables
    def predict(self, observation, action):
        '''
        Predict an action based on an observation.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        :param action: (tensorflow.Tensor) A tensorflow tensor that produces
                                           values that are acceptable to the
                                           observation space.
        '''
        if isinstance(action, dict):
            action = tf.concat(list(action.values()), -1)
        combined_obs_act = tf.concat([observation, action], -1)
        return self.running_network(combined_obs_act)

    @snt.reuse_variables
    def predict_target(self, observation, action):
        '''
        Predict an action based on an observation using the target network.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        :param action: (tensorflow.Tensor) A tensorflow tensor that produces
                                           values that are acceptable to the
                                           observation space.
        '''
        if isinstance(action, dict):
            action = tf.concat(list(action.values()), -1)
        combined_obs_act = tf.concat([observation, action], -1)
        return self.target_network(combined_obs_act)

    def _build(self, observation, action):
        '''
        Build the policy and return a sampling of the action distribution.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        :param action: (tensorflow.Tensor) A tensorflow tensor that produces
                                           values that are acceptable to the
                                           observation space.
        '''
        predict = self.predict(observation, action)
        predict_target = self.predict_target(observation, action)
        update_target = self.update_target()
        return CriticReturn(predict, predict_target, update_target)
