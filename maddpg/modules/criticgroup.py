'''A module for implementing policy groups.'''

from collections import namedtuple

import sonnet as snt

from maddpg.modules import Critic, Group
from maddpg.common.utils_common import zip_map

CriticGroupFunc = namedtuple('CriticGroupFunc', ['values', 'target_values',
                                                 'update_target'])


class CriticGroup(Group):
    '''A class for grouping critics.'''

    def __init__(self, observation_spaces, action_spaces, shared=False,
                 name=None):
        self.shared = shared
        if shared:
            name = next(iter(observation_spaces.keys()))
            obs_space = observation_spaces[name]
            act_space = action_spaces[name]
            shared_critic = Critic(obs_space, act_space, name='shared_critic')
            critics = {}
            for key, (obs, act) in zip_map(observation_spaces, action_spaces):
                assert obs_space == obs
                assert act_space == act
                critics[key] = shared_critic
        else:
            critics = {key: Critic(obs, act, name=key)
                       for key, (obs, act) in
                       zip_map(observation_spaces, action_spaces)}
        super().__init__(critics, name=name)

    def _build(self, observations, actions):
        '''
        Build the agents in the group.

        '''
        values = {}
        target_values = {}
        update = {}
        critics = self.build_critics(observations, actions)
        for name, critic in critics.items():
            values[name] = critic.predict
            target_values[name] = critic.predict_target
            update[name] = critic.update_target
        return CriticGroupFunc(values, target_values, update)

    @snt.reuse_variables
    def build_critics(self, observations, actions):
        '''
        Build the policies in the group.

        :param observations: (dict) A dictionary of tensors that maps names to
                                    observations.
        :return: (PolicyGroupFunc) A namedtuple containing maps of actions,
                                   target actions, update functions.
        '''
        return {name: critic(obs, act) for name, (critic, obs, act) in
                zip_map(self.group, observations, actions)}
