'''A module that contains the MaddpgModule class.'''

from collections import namedtuple
from functools import partial

import sonnet as snt
import tensorflow as tf

import maddpg.common.tf_util as U
from maddpg.modules import PolicyGroup, CriticGroup
from maddpg.common.utils_common import zip_map


MaddpgFunc = namedtuple('MaddpgFunc', ['actor_optimizer', 'critic_optimizer',
                                       'actor_losses', 'critic_losses',
                                       'actor_predict', 'critic_predict',
                                       'actor_update', 'critic_update'])


def train_critic(running_values, target_values, running_params,
                 learning_rate=1e-2, optimizer=tf.train.AdamOptimizer,
                 grad_norm_clipping=None):
    losses = {name: U.mse(running_values[name] - target_values[name])
              for name in running_values.keys()}
    optimizers = {name: U.minimize_and_clip(optimizer(learning_rate),
                                            loss, running_params[name],
                                            grad_norm_clipping)
                  for name, loss in losses.items()}
    return optimizers, losses


def train_actor(values, actor_params, learning_rate=1e-2,
                optimizer=tf.train.AdamOptimizer, grad_norm_clipping=None):
    losses = {name: -tf.reduce_mean(value) for name, value in values.items()}
    optimizers = {name: U.minimize_and_clip(optimizer(learning_rate),
                                            loss, actor_params[name],
                                            grad_norm_clipping)
                  for name, loss in losses.items()}
    return optimizers, losses


def compute_qvalue(reward, done, value, gamma):
    return reward + gamma * (1. - done) * value


class MaddpgModule(snt.AbstractModule):
    '''
    A class for implementing MADDPG.
    '''

    def __init__(self, observation_spaces, action_spaces, shared_policy=False,
                 shared_critic=False, name='maddpg'):
        name = 'maddpg_module' if name is None else name
        super().__init__(name=name)
        self.policy_group = PolicyGroup(observation_spaces, action_spaces,
                                        shared=shared_policy)
        self.critic_group = CriticGroup(observation_spaces, action_spaces,
                                        shared=shared_critic)
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        print('Created module')

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
        qactions = self.policy_group(observation_n).actions
        qvalue = self.compute_qvalue(observation_n, qactions, rewards, dones,
                                     gamma)

        actions = U.concat_map(actions)
        actions = {name: actions for name in qvalue}
        values = self.critic_group(observation, actions).values

        critic_opts, critic_losses = self.get_critic_optimizer(values, qvalue)

        policies = self.policy_group(observation)
        predict = policies.actions
        actions = U.concat_map(predict)
        actions = {name: actions for name in qvalue}
        target_values = self.critic_group(observation, actions).target_values

        policy_opts, policy_losses = self.get_policy_optimizer(target_values,
                                                               policies.entropy)

        return MaddpgFunc(policy_opts, critic_opts,
                          policy_losses, critic_losses,
                          predict, target_values,
                          self.policy_group.update_targets(),
                          self.critic_group.update_targets())

    @snt.reuse_variables
    def compute_qvalue(self, observations, actions, rewards, dones, gamma):
        '''Compute the Q value.'''
        #actions = self.policy_group(observations).actions
        # print(predict)
        _, acts = list(zip(*sorted(actions.items(), key=lambda x: x[0])))
        acts = tf.concat(acts, -1)
        actions = {name: acts for name in actions}
        target = self.critic_group(observations, actions).target_values
        return {name: tf.stop_gradient(R + gamma * (1. - D) * Q)
                for name, (Q, R, D) in zip_map(target, rewards, dones)}

    def get_critic_optimizer(self, values, qvalues):
        '''Get the optimizer for the CriticGroup.'''
        return self.critic_group.create_optimizers(values, qvalues)

    def get_policy_optimizer(self, values, entropy):
        '''Get the optimizer for the PolicyGroup.'''
        return self.policy_group.create_optimizers(values, entropy)
