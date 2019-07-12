'''A module that contains the Policy class.'''

from collections import namedtuple

import numpy as np
import sonnet as snt
import tensorflow as tf

from gym.spaces import Box

import maddpg.common.tf_util as U
from maddpg.common.distributions import make_pdtype
from maddpg.modules.laggingnetwork import LaggingNetwork

PolicyReturn = namedtuple('PolicyReturn', ['predict', 'predict_target',
                                           'update_target', 'entropy'])


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
        self.distribution_type = make_pdtype(action_space)
        out_size = int(self.distribution_type.param_shape()[0])
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
        act_probability = self.distribution_type.pdfromflat(predict)
        running_action = act_probability.sample()
        entropy = act_probability.entropy()
        #entropy = tf.reduce_mean(tf.square(act_probability.flatparam()),
        #                         axis=-1, keepdims=True)
        act_probability = self.distribution_type.pdfromflat(predict_target)
        target_action = act_probability.sample()
        if isinstance(self.action_space, Box):
            low = np.min(self.action_space.low)
            high = np.max(self.action_space.high)
            running_action = tf.clip_by_value(running_action, low, high)
            target_action = tf.clip_by_value(target_action, low, high)
        return PolicyReturn(running_action, target_action, update, entropy)

    @snt.reuse_variables
    def create_optimizer(self, value, entropy=None, learning_rate=1e-3,
                         optimizer=tf.train.AdamOptimizer,
                         grad_norm_clipping=None):
        '''Create an optimizer for the policy.'''
        entropy = 0 if entropy is None else entropy
        params = self.get_trainable_variables()
        loss = -tf.reduce_mean(value) # + tf.reduce_mean(entropy) * 1e-3
        optimizer = U.minimize_and_clip(optimizer(learning_rate), loss,
                                        params, grad_norm_clipping)
        return optimizer, loss
