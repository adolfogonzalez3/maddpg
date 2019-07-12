'''Module for testing functions in utils_common.'''

import pytest
import tensorflow as tf
import maddpg.common.utils_common as utils
from gym.spaces import Box


def test_map_zip_all_equal():
    '''Test map_zip function.'''
    mappings = [{i: i for i in range(10)} for j in range(10)]
    for ke_y, values in utils.zip_map(*mappings):
        assert len(set(values)) == 1


def test_map_zip_all_ascending():
    '''Test map_zip function.'''
    mappings = [{i: i for i in range(10+j)} for j in range(10)]
    for _, values in utils.zip_map(*mappings):
        assert len(set(values)) == 1


def test_map_zip_all_descending():
    '''Test map_zip function.'''
    mappings = [{i: i for i in range(10-j)} for j in range(10)]
    with pytest.raises(KeyError):
        for _, values in utils.zip_map(*mappings):
            assert len(set(values)) == 1


def test_convert_spaces_to_placeholders():
    '''Test convert_spaces_to_placeholders function.'''
    spaces = {str(i): Box(0, 1, shape=(4,)) for i in range(10)}
    placeholders = utils.convert_spaces_to_placeholders(spaces, False)
    assert spaces.keys() == placeholders.keys()
    for _, (space, placeholder) in utils.zip_map(spaces, placeholders):
        assert list(space.shape) == placeholder.shape.as_list()


class SessionStub:
    def __init__(self):
        ...

    def run(self, ops, feed_dict=None):
        return ops


def test_TfFunction():
    '''Test TfFunction class.'''
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            values = {i: tf.get_variable(name=str(i), initializer=0.)
                      for i in range(10)}
            updates = []
            for key, value in values.items():
                updates.append(value.assign(value + key))
        inputs = {i: tf.placeholder(tf.float32, shape=()) for i in range(10)}
        outputs = {i: ph*i for i, ph in inputs.items()}
        init_op = tf.global_variables_initializer()
    session = tf.Session(graph=graph)
    session.run(init_op)
    multiply_by_two = utils.TfFunction(session, inputs=inputs, outputs=outputs)
    update_vars = utils.TfFunction(session, updates=updates)
    get_vars = utils.TfFunction(session, outputs=values)
    data = {i: i for i in range(10)}
    results = multiply_by_two(data)
    for key, value in results.items():
        assert key**2 == value
    for iteration in range(1, 10):
        update_vars()
        for key, value in get_vars().items():
            assert iteration*key == value
