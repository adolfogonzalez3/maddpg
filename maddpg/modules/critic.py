'''A module that contains the Critic class.'''

from collections import namedtuple

import sonnet as snt
import tensorflow as tf

import maddpg.common.tf_util as U
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
        name = 'critic' if name is None else name
        super().__init__((64, 64, 1), name=name)
        self.observation_space = observation_space
        self.action_space = action_space

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
        return self(observation, action).predict

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
        return self(observation, action).predict_target

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
        if isinstance(action, dict):
            _, actions = list(zip(*sorted(action.items(), key=lambda x: x[0])))
            action = tf.concat(actions, -1)
        combined_obs_act = tf.concat([observation, action], -1)
        predict, predict_target, update = super()._build(combined_obs_act)
        return CriticReturn(predict, predict_target, update)

    @snt.reuse_variables
    def create_optimizer(self, values, target_values, learning_rate=1e-2,
                         optimizer=tf.train.AdamOptimizer,
                         grad_norm_clipping=None):
        '''Create an optimizer for the critic.'''
        print('Values:', values.shape, target_values.shape)
        params = self.get_trainable_variables()
        loss = U.mse(values - target_values)
        optimizer = U.minimize_and_clip(optimizer(learning_rate), loss, params,
                                        grad_norm_clipping)
        return optimizer, tf.square(values - target_values)
