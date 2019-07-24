'''A module that contains the MaddpgModule class.'''

from collections import namedtuple

import sonnet as snt
import tensorflow as tf

import maddpg.common.tf_util as U
from maddpg.modules import PolicyGroup, CriticGroup
from maddpg.common.utils_common import zip_map


Td3Func = namedtuple('Td3Func', ['actor_optimizer', 'critic_optimizer',
                                 'actor_losses', 'critic_losses',
                                 'actor_predict', 'critic_predict',
                                 'actor_update', 'critic_update'])


class MaTD3Module(snt.AbstractModule):
    '''
    A class for implementing MADDPG.
    '''

    def __init__(self, observation_spaces, action_spaces, shared_policy=False,
                 shared_critic=False, normalize=None, name='maddpg'):
        name = 'maddpg_module' if name is None else name
        super().__init__(name=name)
        self.policy_group = PolicyGroup(observation_spaces, action_spaces,
                                        shared=shared_policy)
        self.critic_groups = [CriticGroup(observation_spaces, action_spaces,
                                          shared=shared_critic)
                              for _ in range(2)]
        self.normalize = {}
        if normalize:
            if normalize.get('reward'):
                self.normalize['reward'] = {name: snt.BatchNormV2()
                                            for name in action_spaces}
            if normalize.get('observation'):
                self.normalize['observation'] = {name: snt.BatchNormV2()
                                                 for name in action_spaces}
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def _build(self, observation, actions, rewards, observation_n, dones,
               gamma=0.9):
        '''
        Build the networks needed for the MADDPG.

        :param obs: (tensorflow.Tensor) Tensor of observations.
        :param rewards: (tensorflow.Tensor) Tensor of rewards.
        :param dones: (tensorflow.Tensor) Tensor of boolean like values that
                                          denote whether an episode completed
                                          such that if the ith done in dones
                                          is 1 then the ith step was the last
                                          step.
        :param gamma: (float) The gamma value to use.
        :return: (MaddpgFunc) A tuple of functions used for evaluating
                              and training.
        '''
        if self.normalize.get('observation'):
            observation = {key: norm(obs, False) for key, (obs, norm) in
                           zip_map(observation, self.normalize['observation'])}
            observation_n = {key: norm(obs, False) for key, (obs, norm) in
                             zip_map(observation_n,
                                     self.normalize['observation'])}
        if self.normalize.get('reward'):
            rewards = {key: norm(rew, False) for key, (rew, norm) in
                       zip_map(rewards, self.normalize['reward'])}
        obs_n_concat = U.concat_map(observation_n)
        obs_n_concat = {name: obs_n_concat for name in observation}
        qactions = self.policy_group(observation_n).noisy_target
        qactions = U.concat_map(qactions)
        qactions = {name: qactions for name in observation}
        # qvalues = self.compute_qvalue(observation_n, qactions, rewards,
        #  dones, gamma)
        qvalues = self.compute_qvalue(obs_n_concat, qactions, rewards, dones,
                                      gamma)
        actions = U.concat_map(actions)
        actions = {name: actions for name in self.action_spaces}
        obs_concat = U.concat_map(observation)
        obs_concat = {name: obs_concat for name in observation}
        #values_1 = self.critic_group_1(observation, actions).values
        #values_2 = self.critic_group_2(observation, actions).values
        group_values = [critic_group(obs_concat, actions).values
                        for critic_group in self.critic_groups]
        #cr_opts_1, cr_losses_1 = self.get_critic_optimizer(values_1, qvalue)
        critic_opts, critic_losses = list(zip(*[
            critic_group.create_optimizers(values, qvalues)
            for values, critic_group in zip(group_values, self.critic_groups)
        ]))
        #cr_opts_2, cr_losses_2 = self.get_critic_optimizer(values_2, qvalue)
        policies = self.policy_group(observation)
        predict = policies.actions
        actions = U.concat_map(predict)
        actions = {name: actions for name in self.action_spaces}
        #target_vals = self.critic_group_1(observation, actions).target_values
        primary_critics = self.critic_groups[0]
        target_vals = primary_critics(obs_concat, actions).target_values
        po_opts, po_losses = self.get_policy_optimizer(target_vals,
                                                       policies.entropy)
        critic_opts = tf.group(critic_opts)
        # critic_losses = {name: (cr_1 + cr_2)/2 for name, (cr_1, cr_2) in
        #                 zip_map(*critic_losses)}
        critic_losses = {name: tf.reduce_mean(tf.stack(losses, -1), -1)
                         for name, losses in zip_map(*critic_losses)}
        update_critic = tf.group([critic_group.update_targets(5e-3)
                                  for critic_group in self.critic_groups])
        return Td3Func(po_opts, critic_opts, po_losses, critic_losses,
                       predict, target_vals,
                       self.policy_group.update_targets(5e-3),
                       update_critic)

    @snt.reuse_variables
    def compute_qvalue(self, observations, actions, rewards, dones, gamma):
        '''Compute the Q value.'''
        targets = [critic_group(observations, actions).target_values
                   for critic_group in self.critic_groups]
        # target = {name: tf.minimum(t_1, t_2)
        #          for name, (t_1, t_2) in zip_map(*targets)}
        target = {name: tf.reduce_min(tf.stack(target, -1), -1)
                  for name, target in zip_map(*targets)}
        return {name: tf.stop_gradient(R + gamma * (1. - D) * Q)
                for name, (Q, R, D) in zip_map(target, rewards, dones)}

    def get_critic_optimizer(self, group_values, qvalues):
        '''Get the optimizer for the CriticGroup.'''
        return list(zip(*[
            critic_group.create_optimizers(values, qvalues)
            for values, critic_group in zip(group_values, self.critic_groups)
        ]))

    def get_policy_optimizer(self, values, entropy):
        '''Get the optimizer for the PolicyGroup.'''
        return self.policy_group.create_optimizers(values, entropy)
