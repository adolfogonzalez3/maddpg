'''A module that contains the Policy class.'''

from collections import namedtuple

import sonnet as snt

from maddpg.common.distributions import make_pdtype
from maddpg.modules.laggingnetwork import LaggingNetwork

PolicyReturn = namedtuple('PolicyReturn', ['predict', 'predict_target',
                                           'update_target'])
PolicyParams = namedtuple('PolicyParams', ['running', 'target'])


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

    @snt.reuse_variables
    def predict(self, observation):
        '''
        Predict an action based on an observation.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        '''
        logits = self.running_network(observation)
        act_probability = self.distribution_type.pdfromflat(logits)
        return act_probability.sample()

    @snt.reuse_variables
    def predict_target(self, observation):
        '''
        Predict an action based on an observation using the target network.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        '''
        logits = self.target_network(observation)
        act_probability = self.distribution_type.pdfromflat(logits)
        return act_probability.sample()

    def _build(self, observation):
        '''
        Build the policy and return a sampling of the action distribution.

        :param observation: (tensorflow.Tensor) A tensorflow tensor that
                                                produces values that are
                                                acceptable to the observation
                                                space.
        '''
        predict = self.predict(observation)
        predict_target = self.predict_target(observation)
        update_target = self.update_target()
        return PolicyReturn(predict, predict_target, update_target)
