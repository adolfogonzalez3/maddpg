'''A module for testing group class.'''

from gym.spaces import Box
import tensorflow as tf

from maddpg.modules import Policy


def test_build():
    '''Test _build method of Critic.'''
    policy_module = Policy(Box(0, 1, shape=(4,)), Box(0, 1, shape=(2,)))
    observation = tf.placeholder(tf.float32, shape=(None, 4))
    policy = policy_module(observation)
    assert len(policy) == 3
    assert policy.predict.shape.as_list() == [None, 2]
    assert policy.predict_target.shape.as_list() == [None, 2]
    assert isinstance(policy.update_target, tf.Operation)


def test_predict():
    '''Test predict method of Critic.'''
    policy_module = Policy(Box(0, 1, shape=(4,)), Box(0, 1, shape=(2,)))
    observation = tf.placeholder(tf.float32, shape=(None, 4))
    predict_op = policy_module.predict(observation)
    assert predict_op.shape.as_list() == [None, 2]


def test_predict_target():
    '''Test predict_target method of Critic.'''
    policy_module = Policy(Box(0, 1, shape=(4,)), Box(0, 1, shape=(2,)))
    observation = tf.placeholder(tf.float32, shape=(None, 4))
    predict_target_op = policy_module.predict_target(observation)
    assert predict_target_op.shape.as_list() == [None, 2]
