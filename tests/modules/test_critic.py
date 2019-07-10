'''A module for testing the Critic class.'''

from gym.spaces import Box
import tensorflow as tf

from maddpg.modules import Critic


def test_build():
    '''Test _build method of Critic.'''
    critic_module = Critic(Box(0, 1, shape=(4,)), Box(0, 1, shape=(2,)))
    observation = tf.placeholder(tf.float32, shape=(None, 4))
    action = tf.placeholder(tf.float32, shape=(None, 2))
    critic = critic_module(observation, action)
    assert len(critic) == 3
    assert critic.predict.shape.as_list() == [None, 1]
    assert critic.predict_target.shape.as_list() == [None, 1]
    assert isinstance(critic.update_target, tf.Operation)


def test_predict():
    '''Test predict method of Critic.'''
    critic_module = Critic(Box(0, 1, shape=(4,)), Box(0, 1, shape=(2,)))
    observation = tf.placeholder(tf.float32, shape=(None, 4))
    action = tf.placeholder(tf.float32, shape=(None, 2))
    predict_op = critic_module.predict(observation, action)
    assert predict_op.shape.as_list() == [None, 1]


def test_predict_target():
    '''Test predict_target method of Critic.'''
    critic_module = Critic(Box(0, 1, shape=(4,)), Box(0, 1, shape=(2,)))
    observation = tf.placeholder(tf.float32, shape=(None, 4))
    action = tf.placeholder(tf.float32, shape=(None, 2))
    predict_target_op = critic_module.predict_target(observation, action)
    assert predict_target_op.shape.as_list() == [None, 1]
