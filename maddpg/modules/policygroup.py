'''A module for implementing policy groups.'''

from collections import namedtuple

import sonnet as snt
import tensorflow as tf


from maddpg.modules import Policy, Group
from maddpg.common.utils_common import zip_map

PolicyGroupFunc = namedtuple('PolicyGroupFunc', ['policies', 'actions',
                                                 'target_actions',
                                                 'update_target',
                                                 'entropy',
                                                 'noisy_target'])


class PolicyGroup(Group):
    '''A class for grouping policies.'''

    def __init__(self, observation_spaces, action_spaces, shared=False,
                 hyperparameters=None, name=None):
        name = 'policy_group' if name is None else name
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.shared = next(iter(observation_spaces.keys())) if shared else None
        if shared:
            shared_obs_space = observation_spaces[self.shared]
            shared_act_space = action_spaces[self.shared]
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
        super().__init__(policies, shared=self.shared, name=name)

    def _build(self, observations):
        '''
        Build the agents in the group.

        :param observations: (dict) A dictionary of tensors that maps names to
                                    observations.
        :return: (PolicyGroupFunc) A namedtuple containing maps of actions,
                                   target actions, update functions.
        '''

        if self.shared:
            names, observations = list(zip(*list(observations.items())))
            length = len(observations)
            observations = tf.concat(observations, 0)
            policy = self.group[self.shared](observations)
            policies = {name: policy for name in names}
            actions = tf.split(policy.predict, length)
            target_actions = tf.split(policy.predict_target, length)
            entropy = tf.split(policy.entropy, length)
            noisy_target = tf.split(policy.noisy_target, length)
            actions = {name: action for name, action in zip(names, actions)}
            target_actions = {name: action
                              for name, action in zip(names, target_actions)}
            entropy = {name: ent for name, ent in zip(names, entropy)}
            noisy_target = {name: target
                            for name, target in zip(names, noisy_target)}
            update = policy.update_target
        else:
            policies = {name: policy(obs) for name, (policy, obs) in
                        zip_map(self.group, observations)}
            actions = {}
            target_actions = {}
            entropy = {}
            noisy_target = {}
            update = []
            policies = {name: policy(obs) for name, (policy, obs) in
                        zip_map(self.group, observations)}
            for name, policy in policies.items():
                actions[name] = policy.predict
                target_actions[name] = policy.predict_target
                entropy[name] = policy.entropy
                noisy_target[name] = policy.noisy_target
                update.append(policy.update_target)
            update = tf.group(*update)
        return PolicyGroupFunc(policies, actions, target_actions, update,
                               entropy, noisy_target)

    def build_policies(self, observations):
        '''
        Build the policies in the group.

        :param observations: (dict) A dictionary of tensors that maps names to
                                    observations.
        :return: (PolicyGroupFunc) A namedtuple containing maps of actions,
                                   target actions, update functions.
        '''
        return self(observations).policies

    @snt.reuse_variables
    def build_placeholders(self, policies):
        '''
        A helper method for building the placeholders.

        :param policies: (dict) A dictionary mapping strings to Policy objects.
        :return: ((dict, dict)) A tuple of two dictionaries mapping strings
                                to placeholders.
        '''
        phd = {}
        for name, policy in policies.items():
            target = policy.predict_target
            phd[name] = tf.placeholder_with_default(target, shape=target.shape)
        action_trains = {}
        for name, policy in policies.items():
            action_train = phd.copy()
            action_train[name] = policy.predict
            _, actions = list(zip(*sorted(action_train.items(),
                                          key=lambda x: x[0])))
            actions = tf.concat(actions, -1)
            action_trains[name] = actions
        _, phd = list(zip(*sorted(phd.items(), key=lambda x: x[0])))
        phd = tf.concat(phd, -1)
        phd = {name: phd for name in policies}
        return phd, action_trains

    def create_optimizers(self, values):
        '''Create optimizers from the group.'''
        losses = {}
        opts = {}
        learning_rate = self.hyperparameters.get('learning_rate', 1e-4)
        if self.shared:
            policy = self.group[self.shared]
            values = values[self.shared]
            opts, loss = policy.create_optimizer(
                values, learning_rate=learning_rate
            )
            losses = {name: loss for name in self.group}
        else:
            for name, (policy, value) in zip_map(self.group, values):
                opts[name], losses[name] = policy.create_optimizer(
                    value, learning_rate=learning_rate
                )
            opts = tf.group(*list(opts.values()))
        return opts, losses
