
import pytest
import numpy as np
import numpy.random as npr
import tensorflow as tf
from gym import spaces

from maddpg.modules.critic import Critic

SHAPES = ((5,),)


@pytest.mark.parametrize("shape", SHAPES)
def test_critic_predict_shape(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    critic = Critic(observation_space)
    predict = critic.predict(placeholder, placeholder)
    tensor_shape = predict.shape
    assert tensor_shape.rank == len(batch_shape)
    assert tensor_shape.as_list() == [None, 1]


@pytest.mark.parametrize("shape", SHAPES)
def test_critic_predict_run(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    critic = Critic(observation_space)
    predict = critic.predict(placeholder, placeholder)
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        obs = npr.rand(10, *shape)
        action = session.run(predict, feed_dict={placeholder: obs})
        assert action.shape == (10, 1)


@pytest.mark.parametrize("shape", SHAPES)
def test_critic_predict_target_shape(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    critic = Critic(observation_space)
    predict = critic.predict_target(placeholder, placeholder)
    tensor_shape = predict.shape
    assert tensor_shape.rank == len(batch_shape)
    assert tensor_shape.as_list() == [None, 1]


@pytest.mark.parametrize("shape", SHAPES)
def test_critic_predict_target_run(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    critic = Critic(observation_space)
    predict = critic.predict_target(placeholder, placeholder)
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        obs = npr.rand(10, *shape)
        action = session.run(predict, feed_dict={placeholder: obs})
        assert action.shape == (10, 1)


@pytest.mark.parametrize("shape", SHAPES)
def test_critic_update_run(shape):
    batch_shape = [None] + list(shape)
    placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    observation_space = spaces.Box(low=0, high=1, shape=shape)
    critic = Critic(observation_space)
    _ = critic(placeholder, placeholder)
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        session.run(critic.update_target(0.))
        trainable_vars = critic.get_trainable_variables()
        running = session.run(trainable_vars.running)
        target = session.run(trainable_vars.target)
        assert all(np.all(np.isclose(t, r)) for t, r in zip(target, running))
