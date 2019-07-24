'''A module that contains the MaddpgModule class.'''

from collections import namedtuple

import sonnet as snt
import tensorflow as tf

import maddpg.common.tf_util as U
from maddpg.modules import PolicyGroup, CriticGroup
from maddpg.common.utils_common import zip_map

MaddpgFunc = namedtuple('MaddpgFunc', ['actor_optimizer', 'critic_optimizer',
                                       'actor_losses', 'critic_losses',
                                       'actor_predict', 'critic_predict',
                                       'actor_update', 'critic_update'])


class MaddpgModule(snt.AbstractModule):
    '''
    A class for implementing MADDPG.
    '''

    def __init__(self, observation_spaces, action_spaces, shared_policy=False,
                 shared_critic=False, hyperparameters=None, name=None):
        name = 'maddpg_module' if name is None else name
        super().__init__(name=name)
        hyperparameters = hyperparameters if hyperparameters else {}
        self.policy_group = PolicyGroup(
            observation_spaces, action_spaces, shared=shared_policy,
            hyperparameters=hyperparameters.get('policy', {}).copy()
        )
        self.critic_group = CriticGroup(
            observation_spaces, action_spaces, shared=shared_critic,
            hyperparameters=hyperparameters.get('critic', {}).copy()
        )
        self.normalize = hyperparameters.get('normalize', {}).copy()
        if self.normalize:
            if self.normalize.get('reward'):
                self.normalize['reward'] = {
                    name: snt.BatchNormV2(decay_rate=1.0-1e-4)
                    for name in action_spaces
                }
            if self.normalize.get('observation'):
                self.normalize['observation'] = {
                    name: snt.BatchNormV2(decay_rate=1.0-1e-4)
                    for name in action_spaces
                }
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
        qactions = self.policy_group(observation_n).target_actions
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
        #values = self.critic_group(observation, actions).values
        values = self.critic_group(obs_concat, actions).values
        #cr_opts, cr_losses = self.get_critic_optimizer(values, qvalue)
        critic_opts, critic_losses = self.critic_group.create_optimizers(
            values, qvalues
        )
        predict = self.policy_group(observation).actions
        actions = U.concat_map(predict)
        actions = {name: actions for name in self.action_spaces}
        #target_vals = self.critic_group(observation, actions).target_values
        target_vals = self.critic_group(obs_concat, actions).target_values
        po_opts, po_losses = self.get_policy_optimizer(target_vals)
        update_critic = self.critic_group.update_targets(5e-3)
        return MaddpgFunc(po_opts, critic_opts, po_losses, critic_losses,
                          predict, target_vals,
                          self.policy_group.update_targets(5e-3),
                          update_critic)

    @snt.reuse_variables
    def compute_qvalue(self, observations, actions, rewards, dones, gamma):
        '''Compute the Q value.'''
        target = self.critic_group(observations, actions).target_values
        return {name: tf.stop_gradient(R + gamma * (1. - D) * Q)
                for name, (Q, R, D) in zip_map(target, rewards, dones)}

    def get_critic_optimizer(self, values, qvalues):
        '''Get the optimizer for the CriticGroup.'''
        return self.critic_group.create_optimizers(values, qvalues)

    def get_policy_optimizer(self, values):
        '''Get the optimizer for the PolicyGroup.'''
        return self.policy_group.create_optimizers(values)


class MaddpgModuleInference(snt.AbstractModule):
    '''
    A class for implementing MADDPG.
    '''

    def __init__(self, observation_spaces, action_spaces, shared_policy=False,
                 normalize=None, name=None):
        name = 'maddpg_inference_module' if name is None else name
        super().__init__(name=name)
        self.policy_group = PolicyGroup(observation_spaces, action_spaces,
                                        shared=shared_policy)
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
        return MaddpgFunc(None, None, None, None, predict, None, None, None)
