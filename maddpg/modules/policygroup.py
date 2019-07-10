'''A module for implementing policy groups.'''

from collections import namedtuple

import tensorflow as tf
import sonnet as snt

from maddpg.modules import Policy, Group
from maddpg.common.utils_common import zip_map

PolicyGroupFunc = namedtuple('PolicyGroupFunc', ['actions', 'target_actions',
                                                 'update_target'])


class PolicyGroup(Group):
    '''A class for grouping policies.'''

    def __init__(self, observation_spaces, action_spaces, shared=False,
                 name=None):
        self.shared = shared
        if shared:
            name = next(iter(observation_spaces.keys()))
            shared_obs_space = observation_spaces[name]
            shared_act_space = action_spaces[name]
            shared_policy = Policy(shared_obs_space, shared_act_space,
                                   name='shared_policy')
            policies = {}
            for pname, (obs_space, act_space) in zip_map(observation_spaces,
                                                         action_spaces):
                assert shared_obs_space == obs_space
                assert shared_act_space == act_space
                policies[pname] = shared_policy
        else:
            policies = {name: Policy(obs_space, act_space, name=name)
                        for name, (obs_space, act_space) in
                        zip_map(observation_spaces, action_spaces)}
        super().__init__(policies, name=name)

    def _build(self, observations):
        '''
        Build the agents in the group.

        :param observations: (dict) A dictionary of tensors that maps names to
                                    observations.
        :return: (PolicyGroupFunc) A namedtuple containing maps of actions,
                                   target actions, update functions.
        '''
        actions = {}
        target_actions = {}
        update = {}
        for name, policy in self.build_policies(observations).items():
            actions[name] = policy.predict
            target_actions[name] = policy.predict_target
            update[name] = policy.update_target
        return PolicyGroupFunc(actions, target_actions, update)

    @snt.reuse_variables
    def build_policies(self, observations):
        '''
        Build the policies in the group.

        :param observations: (dict) A dictionary of tensors that maps names to
                                    observations.
        :return: (PolicyGroupFunc) A namedtuple containing maps of actions,
                                   target actions, update functions.
        '''
        return {name: policy(obs) for name, (policy, obs) in
                zip_map(self.group, observations)}

    @snt.reuse_variables
    def build_placeholders(self, observations):
        '''
        Build placeholders that default to target actions.

        :param observations: (dict) A dictionary of tensors that maps names to
                                    observations.
        :return: (dict) A mapping of names to placeholders.
        '''
        phd = {}
        policies = self.build_policies(observations)
        for name, policy in policies.items():
            target = policy.predict_target
            phd[name] = tf.placeholder_with_default(target, shape=target.shape)
        action_trains = {}
        for name, policy in policies.items():
            action_train = phd.copy()
            action_train[name] = policy.predict
            action_trains[name] = action_train
        phd = {name: phd for name in phd}
        return phd, action_trains
