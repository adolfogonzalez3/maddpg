'''A module that contains the Policy class.'''

from collections import namedtuple

import sonnet as snt
import tensorflow as tf

LaggingNetReturn = namedtuple('LaggingNetReturn', ['predict', 'predict_target',
                                                   'update_target'])


class LaggingNetwork(snt.AbstractModule):
    '''
    A class for implementing a lagging network.
    '''

    def __init__(self, node_sizes, name='lagging_network'):
        super().__init__(name=name)
        self.running_network = snt.nets.MLP(node_sizes, name='running')
        self.target_network = snt.nets.MLP(node_sizes, name='target')

    def _build(self, inputs):
        '''
        Build the networks.

        :param inputs: (tensorflow.Tensor) A tensor.
        :return: (tuple) Returns a namedtuple containing the running and target
                         tensors.
        '''
        running_predict = self.running_network(inputs)
        target_predict = self.target_network(inputs)
        update = self.update_target()
        return LaggingNetReturn(running_predict, target_predict, update)

    @snt.reuse_variables
    def update_target(self, polyak=None):
        '''
        Update the target network.
        '''
        polyak = 1.0 - 1e-2 if polyak is None else polyak
        rvars = self.running_network.trainable_variables
        rvars = sorted(rvars, key=lambda v: v.name)
        tvars = self.target_network.trainable_variables
        tvars = sorted(tvars, key=lambda v: v.name)
        expression = [tvar.assign(polyak * tvar + (1.0-polyak)*rvar)
                      for rvar, tvar in zip(rvars, tvars)]
        return tf.group(*expression, name='update_target')

    def get_trainable_variables(self):
        '''Retrieve the trainable variables of the policy.'''
        return self.running_network.trainable_variables
