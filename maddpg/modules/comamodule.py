'''A module that contains the MaddpgModule class.'''

import sys
from collections import namedtuple

import sonnet as snt
import tensorflow as tf

import maddpg.common.tf_util as U
from maddpg.modules import PolicyGroup, CriticGroup
from maddpg.common.utils_common import zip_map


ComaFunc = namedtuple('ComaFunc', ['actor_optimizer', 'critic_optimizer',
                                   'actor_losses', 'critic_losses',
                                   'actor_predict', 'critic_predict',
                                   'actor_update', 'critic_update'])


class ComaModule(snt.AbstractModule):
    '''
    A class for implementing MADDPG.
    '''

    def __init__(self, observation_spaces, action_spaces, shared_policy=False,
                 shared_critic=False, normalize=None, name='maddpg'):
        name = 'maddpg_module' if name is None else name
        super().__init__(name=name)
        self.best_policy_group = PolicyGroup(observation_spaces, action_spaces,
                                             shared=shared_policy,
                                             name='best_policy')
        self.worst_policy_group = PolicyGroup(observation_spaces,
                                              action_spaces,
                                              shared=shared_policy,
                                              name='worst_policy')
        self.global_critic_group = CriticGroup(observation_spaces,
                                               action_spaces,
                                               shared=True,
                                               name='global_critic')
        self.personal_critic_group = CriticGroup(observation_spaces,
                                                 action_spaces,
                                                 shared=False,
                                                 name='personal_critic')
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
               gamma=0.95):
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
        global_critics = self.global_critic_group
        worst_qactions = self.worst_policy_group(observation_n).actions
        worst_qactions = U.concat_map(worst_qactions)
        worst_qactions = {name: worst_qactions for name in observation}
        worst_qvalues = global_critics(obs_n_concat,
                                       worst_qactions).target_values

        best_qactions = self.best_policy_group(observation_n).actions
        best_qactions = U.concat_map(best_qactions)
        best_qactions = {name: best_qactions for name in observation}
        best_qvalues = self.compute_global_qvalue(obs_n_concat, best_qactions,
                                                  rewards, dones, gamma)
        all_actions = U.concat_map(actions)
        all_actions = {name: all_actions for name in self.action_spaces}
        obs_concat = U.concat_map(observation)
        obs_concat = {name: obs_concat for name in observation}

        global_values = global_critics(obs_concat, all_actions).values

        global_opts = global_critics.create_optimizers(global_values,
                                                       best_qvalues)

        personal_reward = {
            name: tf.stop_gradient(gval - wval)
            for name, (gval, wval) in zip_map(global_values, worst_qvalues)
        }
        personal_critics = self.personal_critic_group
        personal_values = personal_critics(obs_concat, all_actions).values
        personal_qvalue = self.compute_personal_qvalue(obs_n_concat,
                                                       best_qactions,
                                                       personal_reward,
                                                       dones, gamma)
        personal_critic = personal_critics.create_optimizers(personal_values,
                                                             personal_qvalue)

        predict = self.best_policy_group(observation).actions
        all_actions = U.concat_map(predict)
        all_actions = {name: all_actions for name in self.action_spaces}
        target_vals = personal_critics(obs_concat, all_actions).target_values

        worst_predict = self.worst_policy_group(observation).actions
        worst_predict = U.concat_map(worst_predict)
        worst_predict = {name: worst_predict for name in self.action_spaces}
        worst_vals = personal_critics(obs_concat, worst_predict).target_values
        worst_vals = {name: -v for name, v in worst_vals.items()}

        best_policy = self.best_policy_group.create_optimizers(target_vals)
        worst_policy = self.worst_policy_group.create_optimizers(worst_vals)

        critic_opts = [global_opts[0], personal_critic[0]]
        critic_losses = [global_opts[1], personal_critic[1]]
        #po_opts = [best_policy[0], worst_policy[0]]
        #po_losses = [best_policy[1], worst_policy[1]]
        po_opts = [best_policy[0], worst_policy[0]]
        po_losses = [best_policy[1], worst_policy[1]]

        critic_opts = tf.group(critic_opts)
        critic_losses = {name: tf.reduce_mean(tf.stack(losses, -1), -1)
                         for name, losses in zip_map(*critic_losses)}
        update_critic = tf.group([
            global_critics.update_targets(5e-3),
            self.personal_critic_group.update_targets(5e-3)
        ])

        po_opts = tf.group(po_opts)
        po_losses = {name: tf.math.reduce_std(tf.stack(losses, -1), -1)
                     for name, losses in zip_map(*po_losses)}
        update_policy = tf.group([
            self.worst_policy_group.update_targets(5e-3),
            self.best_policy_group.update_targets(5e-3)
        ])

        return ComaFunc(po_opts, critic_opts, po_losses, critic_losses,
                        predict, target_vals, update_policy, update_critic)

    @snt.reuse_variables
    def compute_global_qvalue(self, observations, actions, rewards, dones,
                              gamma):
        '''Compute the Q value.'''
        targets = self.global_critic_group(observations, actions).target_values
        return {name: tf.stop_gradient(R + gamma * (1. - D) * Q)
                for name, (Q, R, D) in zip_map(targets, rewards, dones)}

    @snt.reuse_variables
    def compute_personal_qvalue(self, observations, actions, rewards, dones,
                                gamma):
        '''Compute the Q value.'''
        targets = self.personal_critic_group(observations,
                                             actions).target_values
        return {name: tf.stop_gradient(R + gamma * (1. - D) * Q)
                for name, (Q, R, D) in zip_map(targets, rewards, dones)}


class ComaModuleInference(snt.AbstractModule):
    '''
    A class for implementing MADDPG.
    '''

    def __init__(self, observation_spaces, action_spaces, shared_policy=False,
                 normalize=None, name=None):
        name = 'maddpg_inference_module' if name is None else name
        super().__init__(name=name)
        self.policy_group = PolicyGroup(observation_spaces, action_spaces,
                                        shared=shared_policy,
                                        name='best_policy')
        self.normalize = {}
        if normalize:
            if normalize.get('observation'):
                self.normalize['observation'] = {name: snt.BatchNormV2()
                                                 for name in action_spaces}
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def _build(self, observation):
        '''
        Build the networks needed for the MADDPG.

        :param obs: (tensorflow.Tensor) Tensor of observations.
        :return: (MaddpgFunc) A tuple of functions used for evaluating
                              and training.
        '''
        if self.normalize.get('observation'):
            observation = {key: norm(obs, False) for key, (obs, norm) in
                           zip_map(observation, self.normalize['observation'])}

        policies = self.policy_group(observation)
        predict = policies.actions
        return ComaFunc(None, None, None, None, predict, None, None, None)
