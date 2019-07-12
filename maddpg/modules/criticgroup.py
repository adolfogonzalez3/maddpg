'''A module for implementing policy groups.'''

from collections import namedtuple

import tensorflow as tf

from maddpg.modules import Critic, Group
from maddpg.common.utils_common import zip_map

CriticGroupFunc = namedtuple('CriticGroupFunc', ['critics', 'values',
                                                 'target_values',
                                                 'update_target'])


class CriticGroup(Group):
    '''A class for grouping critics.'''

    def __init__(self, observation_spaces, action_spaces, shared=False,
                 name=None):
        name = 'critic_group' if name is None else name
        self.shared = next(iter(observation_spaces.keys())) if shared else None
        if shared:
            #name = next(iter(observation_spaces.keys()))
            obs_space = observation_spaces[self.shared]
            act_space = action_spaces[self.shared]
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
        super().__init__(critics, shared=self.shared, name=name)

    def _build(self, observations, actions):
        '''
        Build the agents in the group.

        '''
        if self.shared:
            observations = sorted(list(observations.items()),
                                  key=lambda x: x[0])
            actions = sorted(list(actions.items()),
                             key=lambda x: x[0])
            names, observations = list(zip(*observations))
            _, actions = list(zip(*actions))
            length = len(observations)
            observations = tf.concat(observations, 0)
            actions = tf.concat(actions, 0)
            critic = self.group[self.shared](observations, actions)
            critics = {name: critic for name in names}
            values = tf.split(critic.predict, length)
            target_values = tf.split(critic.predict_target, length)
            values = {name: value for name, value in zip(names, values)}
            target_values = {name: value
                             for name, value in zip(names, target_values)}
            #update = {name: policy.update_target for name in self.group}
            update = critic.update_target
        else:
            critics = {name: critic(obs, act) for name, (critic, obs, act) in
                       zip_map(self.group, observations, actions)}
            values = {}
            target_values = {}
            update = []
            for name, critic in critics.items():
                values[name] = critic.predict
                target_values[name] = critic.predict_target
                update.append(critic.update_target)
            update = tf.group(*update)
        return CriticGroupFunc(critics, values, target_values, update)

    def build_critics(self, observations, actions):
        '''
        Build the policies in the group.

        :param observations: (dict) A dictionary of tensors that maps names to
                                    observations.
        :return: (PolicyGroupFunc) A namedtuple containing maps of actions,
                                   target actions, update functions.
        '''
        return self(observations, actions).critics

    def create_optimizers(self, values, qvalues):
        '''Create optimizers from the group.'''
        losses = {}
        opts = {}
        if self.shared:
            critic = self.group[self.shared]
            values = values[self.shared]
            qvalues = qvalues[self.shared]
            opts, loss = critic.create_optimizer(values, qvalues)
            losses = {name: loss for name in self.group}
        else:
            for name, (critic, value, Q) in zip_map(self.group, values,
                                                    qvalues):
                opts[name], losses[name] = critic.create_optimizer(value, Q)
            opts = tf.group(*list(opts.values()))
        return opts, losses
