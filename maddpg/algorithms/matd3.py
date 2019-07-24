'''A module that contains the Maddpg class.'''
from collections import namedtuple

from maddpg.modules import MaTD3Module
from maddpg.common.utils_common import TfFunction, flatten_map, unflatten_map
from maddpg.algorithms import MultiAgentAlgBase

SharedOptions = namedtuple('SharedOptions', ['policy', 'critic'])


class MaTd3(MultiAgentAlgBase):
    '''
    A class for implementing mutiple agent TD3.
    '''

    def __init__(self, observation_space, action_space, shared_policy=False,
                 shared_critic=False, normalize=None):
        super().__init__(observation_space, action_space)
        observation_spaces = self.observation_space.spaces
        action_spaces = self.action_space.spaces
        self.policy = SharedOptions(shared_policy, shared_critic)
        with self.graph.as_default():
            maddpg_module = MaTD3Module(observation_spaces, action_spaces,
                                         shared_policy, shared_critic,
                                         normalize=normalize)
            functions = maddpg_module(*self.placeholders)
        self._predicts_h = TfFunction(
            self.session, inputs=self.placeholders.observations,
            outputs=functions.actor_predict
        )
        self._compute_values_h = TfFunction(
            self.session, inputs=self.placeholders.observations,
            outputs=functions.critic_predict
        )
        updates = [functions.actor_update, functions.critic_update]
        self._update_targets = TfFunction(self.session, updates=updates)
        optimizers = [functions.actor_optimizer, functions.critic_optimizer]
        inputs = flatten_map(self.placeholders._asdict())
        outputs = flatten_map({'actor': functions.actor_losses,
                               'critic': functions.critic_losses})
        self._train = TfFunction(self.session, inputs=inputs,
                                 updates=optimizers, outputs=outputs)
        self._train_critic = TfFunction(
            self.session, inputs=inputs, updates=[functions.critic_optimizer],
            outputs=functions.critic_losses
        )
        self._compute_loss_h = TfFunction(self.session, inputs=inputs,
                                          outputs=outputs)

    def _predicts(self, observations):
        return self._predicts_h(observations)

    def _compute_values(self, observations):
        return self._compute_values_h(observations)

    def update_targets(self):
        '''Update the target networks for all agents.'''
        self._update_targets()

    def run_updates(self):
        self.update_targets()

    def _train_step(self, observations, actions, rewards, observations_n,
                    dones, step=None):
        '''Train the agents.'''
        feed_dict = self.placeholders._make([observations, actions, rewards,
                                             observations_n, dones])
        feed_dict = flatten_map(feed_dict._asdict())
        if step and step % 2 == 0:
            results = unflatten_map(self._train(feed_dict))
        else:
            results = unflatten_map(self._train_critic(feed_dict))
        return results

    def _compute_loss(self, observations, actions, rewards, observations_n,
                      dones):
        '''Compute the loss of the agents.'''
        feed_dict = self.placeholders._make([observations, actions, rewards,
                                             observations_n, dones])
        feed_dict = flatten_map(feed_dict._asdict())
        return unflatten_map(self._compute_loss_h(feed_dict))
