'''A module for testing the PolicyGroup class.'''

from gym.spaces import Box
import tensorflow as tf

from maddpg.modules import PolicyGroup
from maddpg.common.utils_common import zip_map


def test_build():
    '''Test _build method of PolicyGroup.'''
    observation_spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    action_spaces = {str(i): Box(0, 1, shape=(2,)) for i in range(10)}
    policy_group_module = PolicyGroup(observation_spaces, action_spaces)
    observations = {str(i): tf.placeholder(tf.float32, shape=(None, 4))
                    for i in range(10)}
    policy_group = policy_group_module(observations)
    assert len(policy_group) == 3
    for _, (predict, predict_target, update) in zip_map(*policy_group):
        assert predict.shape.as_list() == [None, 2]
        assert predict_target.shape.as_list() == [None, 2]
        assert isinstance(update, tf.Operation)


def test_build_critics():
    '''Test build_policies method of PolicyGroup.'''
    observation_spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    action_spaces = {str(i): Box(0, 1, shape=(2,)) for i in range(10)}
    policy_group_module = PolicyGroup(observation_spaces, action_spaces)
    observations = {str(i): tf.placeholder(tf.float32, shape=(None, 4))
                    for i in range(10)}
    policies = policy_group_module.build_policies(observations)
    assert len(policies) == 10
    assert policies.keys() == policy_group_module.group.keys()
    for name, policy in policy_group_module.group.items():
        for other_name, other_policy in policy_group_module.group.items():
            if name != other_name:
                assert policy != other_policy


def test_build_critics_shared():
    '''Test build_policies method of PolicyGroup with shared policy.'''
    observation_spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    action_spaces = {str(i): Box(0, 1, shape=(2,)) for i in range(10)}
    policy_group_module = PolicyGroup(observation_spaces, action_spaces,
                                      shared=True)
    observations = {str(i): tf.placeholder(tf.float32, shape=(None, 4))
                    for i in range(10)}
    policies = policy_group_module.build_policies(observations)
    assert len(policies) == 10
    assert policies.keys() == policy_group_module.group.keys()
    policy = next(iter(policy_group_module.group.values()))
    for other_policy in policy_group_module.group.values():
        assert policy == other_policy


def test_build_placeholders():
    '''Test build_placeholders method of PolicyGroup.'''
    observation_spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    action_spaces = {str(i): Box(0, 1, shape=(2,)) for i in range(10)}
    policy_group_module = PolicyGroup(observation_spaces, action_spaces,
                                      shared=True)
    observations = {str(i): tf.placeholder(tf.float32, shape=(None, 4))
                    for i in range(10)}
    phd, phd_train = policy_group_module.build_placeholders(observations)
    assert len(phd) == 10
    assert phd.keys() == policy_group_module.group.keys()
    assert len(phd_train) == 10
    assert phd_train.keys() == policy_group_module.group.keys()
