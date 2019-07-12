'''A module for testing the CriticGroup class.'''

from gym.spaces import Box
import tensorflow as tf

from maddpg.modules import CriticGroup
from maddpg.common.utils_common import zip_map


def test_build():
    '''Test _build method of CriticGroup.'''
    observation_spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    action_spaces = {str(i): Box(0, 1, shape=(2,)) for i in range(10)}
    critic_group_module = CriticGroup(observation_spaces, action_spaces)
    observations = {str(i): tf.placeholder(tf.float32, shape=(None, 4))
                    for i in range(10)}
    actions = {str(i): tf.placeholder(tf.float32, shape=(None, 2))
               for i in range(10)}
    critic_group = critic_group_module(observations, actions)
    assert len(critic_group) == 4
    for value in critic_group.values.values():
        assert value.shape.as_list() == [None, 1]
    for target_value in critic_group.target_values.values():
        assert target_value.shape.as_list() == [None, 1]
    for update in critic_group.update_target.values():
        assert isinstance(update, tf.Operation)


def test_build_critics():
    '''Test build_critics method of CriticGroup.'''
    observation_spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    action_spaces = {str(i): Box(0, 1, shape=(2,)) for i in range(10)}
    critic_group_module = CriticGroup(observation_spaces, action_spaces)
    observations = {str(i): tf.placeholder(tf.float32, shape=(None, 4))
                    for i in range(10)}
    actions = {str(i): tf.placeholder(tf.float32, shape=(None, 2))
               for i in range(10)}
    critics = critic_group_module.build_critics(observations, actions)
    assert len(critics) == 10
    for name, critic in critic_group_module.group.items():
        for other_name, other_critic in critic_group_module.group.items():
            if name != other_name:
                assert critic != other_critic


def test_build_critics_shared():
    '''Test build_critics method of CriticGroup shared.'''
    observation_spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    action_spaces = {str(i): Box(0, 1, shape=(2,)) for i in range(10)}
    critic_group_module = CriticGroup(observation_spaces, action_spaces,
                                      shared=True)
    observations = {str(i): tf.placeholder(tf.float32, shape=(None, 4))
                    for i in range(10)}
    actions = {str(i): tf.placeholder(tf.float32, shape=(None, 2))
               for i in range(10)}
    critics = critic_group_module.build_critics(observations, actions)
    assert len(critics) == 10
    critic = next(iter(critic_group_module.group.values()))
    for other_critic in critic_group_module.group.values():
        assert critic == other_critic
