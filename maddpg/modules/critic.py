'''A module that contains the Critic class.'''

from collections import namedtuple

import tensorflow as tf

from maddpg.modules.laggingnetwork import LaggingNetwork

CriticReturn = namedtuple('PolicyReturn', ['predict', 'predict_target',
                                           'update_target'])
CriticParams = namedtuple('PolicyParams', ['running', 'target'])


class Critic(LaggingNetwork):
    '''
    A class for implementing a critic.
    '''

    def __init__(self, observation_space, name='critic'):
        self.observation_space = observation_space
        super().__init__((64, 64, 1), name=name)

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
        combined_obs_act = tf.concat([observation, action], -1)
        return self.running_network(combined_obs_act)

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

    def get_trainable_variables(self):
        '''Retrieve the trainable variables of the policy.'''
        return CriticParams(self.running_network.trainable_variables,
                            self.target_network.trainable_variables)
