'''A module for testing the Group class.'''

import numpy as np
import sonnet as snt
import tensorflow as tf

from maddpg.modules import Group


class TestMLP(snt.AbstractModule):
    def __init__(self):
        super().__init__(name='test_network')
        self.network = snt.nets.MLP([10])

    def _build(self, inputs):
        return self.network(inputs)

    def get_trainable_variables(self):
        return self.network.trainable_variables


class TestGroup(Group):
    def __init__(self):
        super().__init__({name: TestMLP() for name in range(10)},
                         name='test_group')

    def _build(self):
        inputs = tf.placeholder(tf.float32, shape=(None, 9))
        return {name: net(inputs) for name, net in self.group.items()}


def test_get_trainable_variables():
    '''Test get_trainable_variables method of Group.'''
    group_module = TestGroup()
    group = group_module()
    assert group.keys() == group_module.group.keys()
    for _, variables in group_module.get_trainable_variables().items():
        assert len(variables) == 2
        assert sum(np.prod(v.shape) for v in variables) == 100
