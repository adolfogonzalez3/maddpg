'''A module containing a class for implementing groups.'''

import sonnet as snt


class Group(snt.AbstractModule):
    '''An abstract class for building GroupTypes.'''

    def __init__(self, group, name='group'):
        '''
        Create a group object.

        :param group: ([]) A sequence of modules.
        '''
        super().__init__(name=name)
        self.group = group

    def get_trainable_variables(self):
        '''Retrieve the trainable variables of the group.'''
        return {name: member.get_trainable_variables()
                for name, member in self.group.items()}
