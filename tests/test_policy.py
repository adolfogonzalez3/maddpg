
import pytest
import numpy as np
import numpy.random as npr
import tensorflow as tf
from gym import spaces

from maddpg.modules.policy import Policy

SHAPES = ((5,),)


@pytest.mark.parametrize("shape", SHAPES)
def test_policy_predict_shape(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    action_space = spaces.Box(low=0, high=1, shape=shape)
    policy = Policy(observation_space, action_space)
    predict = policy.predict(placeholder)
    tensor_shape = predict.shape
    assert tensor_shape.rank == len(batch_shape)
    assert tensor_shape.as_list() == batch_shape


@pytest.mark.parametrize("shape", SHAPES)
def test_policy_predict_run(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    action_space = spaces.Box(low=0, high=1, shape=shape)
    policy = Policy(observation_space, action_space)
    predict = policy.predict(placeholder)
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        obs = npr.rand(10, *shape)
        action = session.run(predict, feed_dict={placeholder: obs})
        assert action.shape == (10, *shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_policy_predict_target_shape(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    action_space = spaces.Box(low=0, high=1, shape=shape)
    policy = Policy(observation_space, action_space)
    predict = policy.predict_target(placeholder)
    tensor_shape = predict.shape
    assert tensor_shape.rank == len(batch_shape)
    assert tensor_shape.as_list() == batch_shape


@pytest.mark.parametrize("shape", SHAPES)
def test_policy_predict_target_run(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    action_space = spaces.Box(low=0, high=1, shape=shape)
    policy = Policy(observation_space, action_space)
    predict = policy.predict_target(placeholder)
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        obs = npr.rand(10, *shape)
        action = session.run(predict, feed_dict={placeholder: obs})
        assert action.shape == (10, *shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_policy_update_run(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    action_space = spaces.Box(low=0, high=1, shape=shape)
    policy = Policy(observation_space, action_space)
    _ = policy(placeholder)
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        session.run(policy.update_target(0.))
        trainable_vars = policy.get_trainable_variables()
        running = session.run(trainable_vars.running)
        target = session.run(trainable_vars.target)
        assert all(np.all(np.isclose(t, r)) for t, r in zip(target, running))
