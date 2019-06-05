'''A module that contains the Policy class.'''

from collections import namedtuple

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
        self.distribution_type = make_pdtype(action_space)
        out_size = int(self.distribution_type.param_shape()[0])
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__((64, 64, out_size), name=name)

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

    def get_trainable_variables(self):
        '''Retrieve the trainable variables of the policy.'''
        return PolicyParams(self.running_network.trainable_variables,
                            self.target_network.trainable_variables)
