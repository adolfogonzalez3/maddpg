'''A module that contains the Policy class.'''

from collections import namedtuple

import numpy as np
import sonnet as snt
import tensorflow as tf

from gym.spaces import Box, Discrete

import maddpg.common.tf_util as U
from maddpg.modules.laggingnetwork import LaggingNetwork

PolicyReturn = namedtuple('PolicyReturn', ['predict', 'predict_target',
                                           'update_target', 'entropy',
                                           'noisy_target'])


class Policy(LaggingNetwork):
    '''
    A class for implementing a policy gradient.
    '''

    def __init__(self, observation_space, action_space, name='policy'):
        '''
        Create a policy.

        :param observation_space: (gym.space) A space compatible with gym.
        :param action_space: (gym.space) A space compatible with gym.
        :param name: (str) The name of the policy.
        '''
        out_size = action_space.shape[0]
        super().__init__((64, 64, out_size), name=name)
        self.observation_space = observation_space
        self.action_space = action_space

    def predict(self, observation):
        '''
        Predict an action based on an observation.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        '''
        return self(observation).predict

    def predict_target(self, observation):
        '''
        Predict an action based on an observation using the target network.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        '''
        return self(observation).predict_target

    def _build(self, observation):
        '''
        Build the policy and return a sampling of the action distribution.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        '''
        predict, predict_target, update = super()._build(observation)
        running_action = tf.tanh(predict)
        target_action = tf.tanh(predict_target)
        entropy = running_action
        clipped_noise = tf.random.normal(tf.shape(target_action), stddev=0.2)
        clipped_noise = tf.clip_by_value(clipped_noise, -0.5, 0.5)
        noisy_target = target_action + clipped_noise
        noisy_target = tf.clip_by_value(noisy_target, -1, 1)
        if isinstance(self.action_space, Box):
            low = np.min(self.action_space.low)
            high = np.max(self.action_space.high)
            interval = (high - low) / 2
            adjust = interval + low
            running_action = running_action*interval + adjust
            target_action = target_action*interval + adjust
            noisy_target = noisy_target*interval + adjust
        elif isinstance(self.action_space, Discrete):
            raise RuntimeError()
        else:
            raise RuntimeError()
        return PolicyReturn(running_action, target_action, update, entropy,
                            noisy_target)

    @snt.reuse_variables
    def create_optimizer(self, value, learning_rate=1e-5,
                         optimizer_fn=tf.train.AdamOptimizer,
                         grad_norm_clipping=None):
        '''Create an optimizer for the policy.'''
        params = self.get_trainable_variables()
        loss = -tf.reduce_mean(value)
        optimizer = optimizer_fn(learning_rate, use_locking=True)
        optimizer = U.minimize_and_clip(optimizer, loss, params,
                                        grad_norm_clipping)
        return optimizer, loss
