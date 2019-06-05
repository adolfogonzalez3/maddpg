'''A module that contains the Policy class.'''

from collections import namedtuple

import sonnet as snt
import tensorflow as tf

LaggingNetReturn = namedtuple('LaggingNetReturn', ['predict', 'predict_target',
                                                   'update_target'])
LaggingNetParams = namedtuple('LaggingNetParams', ['running', 'target'])


class LaggingNetwork(snt.AbstractModule):
    '''
    A class for implementing a lagging network.
    '''

    def __init__(self, node_sizes, name='lagging_network'):
        super().__init__(name=name)
        with self._enter_variable_scope():  # This line is crucial!
            self.running_network = snt.nets.MLP(node_sizes,
                                                name='running_function')
            self.target_network = snt.nets.MLP(node_sizes,
                                               name='target_function')

    def predict(self, observation):
        '''
        Predict based on an observation.

        :param observation: (tensorflow.Tensor) A tensorflow tensor.
        '''
        return self.running_network(observation)

    def predict_target(self, observation):
        '''
        Predict based on an observation using the target network.

        :param observation: (tensorflow.Tensor) A tensorflow tensor.
        '''
        return self.target_network(observation)

    def update_target(self, polyak=1.0 - 1e-2):
        '''
        Update the target network.
        '''
        rvars = self.running_network.trainable_variables
        tvars = self.target_network.trainable_variables
        expression = [tvar.assign(polyak * tvar + (1.0-polyak)*rvar)
                      for rvar, tvar in zip(rvars, tvars)]
        return tf.group(*expression, name='update_target')

    def _build(self, observation):
        '''
        Build the policy and return a sampling of the action distribution.

        :param observation: (tensorflow.Tensor) A tensorflow tensor.
        '''
        predict = self.predict(observation)
        predict_target = self.predict_target(observation)
        update_target = self.update_target()
        return LaggingNetReturn(predict, predict_target, update_target)

    def get_trainable_variables(self):
        '''Retrieve the trainable variables of the policy.'''
        return LaggingNetParams(self.running_network.trainable_variables,
                                self.target_network.trainable_variables)
