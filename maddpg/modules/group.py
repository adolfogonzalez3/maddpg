'''A module containing a class for implementing groups.'''

from abc import abstractmethod

import sonnet as snt
import tensorflow as tf


class Group(snt.AbstractModule):
    '''An abstract class for building GroupTypes.'''

    def __init__(self, group, shared=False, name='group'):
        '''
        Create a group object.

        :param group: ([]) A sequence of modules.
        '''
        name = 'group' if name is None else name
        super().__init__(name=name)
        self.group = group
        self.shared = shared

    @abstractmethod
    def _build(self, *args, **kwargs):
        ...

    def get_trainable_variables(self):
        '''Retrieve the trainable variables of the group.'''
        return {name: member.get_trainable_variables()
                for name, member in self.group.items()}

    def update_targets(self):
        '''Update target networks.'''
        if self.shared:
            update = self.group[self.shared].update_target()
        else:
            update = tf.group(*[member.update_target()
                                for member in self.group.values()])
        return update
