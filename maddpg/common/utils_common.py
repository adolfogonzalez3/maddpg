'''A module that contains utilites commonly used throughout the library.'''

from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gym.spaces import Box, Discrete, Dict

# Logging


def print_tqdm(*args):
    '''Print python-style using tqdm.'''
    tqdm.write(' '.join(str(arg) for arg in args))

# Map functions


def zip_map(*args, keys=None):
    '''
    Iterate through all mappings using a sequence of keys.

    :param keys: (None or Sequence) If None then take the first mapping's keys
                                    and use that for iteration else if Sequence
                                    then iterate through all mappings using
                                    the keys in Sequence.
    :yield: ([Object]) A tuple of Objects retrieved from the mappings.
    '''
    keys = args[0].keys() if keys is None else keys
    for key in keys:
        yield (key, [arg[key] for arg in args])


def map_apply(dictionary, function):
    '''
    Apply a function to data in a dictionary.

    :param dictionary: (dict) A dictionary containing data.
    :param function: (callable) A callable that is applied to the data.
    :return: (dict) A new dictionary with results of callable for each data.
    '''
    return {key: function(value) for key, value in dictionary.items()}


def flatten_map(dictionary):
    '''
    Flatten a nested dict by prefixing a subdict's key with its parent.

    '''
    new_map = {}
    for parent, child_dict in dictionary.items():
        for child, value in child_dict.items():
            new_map['{}_{}'.format(parent, child)] = value
    return new_map


def map_to_list(dictionary):
    '''
    Convert a map to a list sorted on the keys.

    :param dictionary: (dict) A dictionary to turn into a list.
    :return: (list) A list of values sorted by their keys.
    '''
    return list(zip(*sorted(dictionary.items(), key=lambda x: x[0])))[1]


def unflatten_map(dictionary):
    '''
    Reverse prior flattening of a dictionary.
    '''
    new_map = defaultdict(dict)
    for key, value in dictionary.items():
        parent, child = key.split('_', maxsplit=1)
        new_map[parent][child] = value
    return dict(new_map)


def map_to_batch(dictionary, shapes):
    '''
    Reshape data in a dictionary to correct batch shapes.

    :param dictionary:
    '''
    return {key: np.reshape(value, (-1,) + shape)
            for key, (value, shape) in zip_map(dictionary, shapes)}

# Gym Environment functions


def convert_spaces_to_placeholders(spaces, batch_dim=True):
    '''
    Convert a map of spaces to placeholders.

    :param spaces: ({gym.Space}) A map of spaces.
    :return: ({tensorflow.placeholder}) A map of placeholders.
    '''
    placeholders = {}
    for name, space in spaces.items():
        if isinstance(space, Box):
            shape = (None,) + space.shape if batch_dim else space.shape
            placeholders[name] = tf.placeholder(tf.float32, shape=shape)
        elif isinstance(space, Discrete):
            shape = (None, space.n) if batch_dim else (space.n,)
            placeholders[name] = tf.placeholder(tf.float32, shape=shape)
        else:
            raise RuntimeError(('Space not currently handled.'
                                ' {!s}'.format(space)))
    return placeholders


def get_shape(space):
    '''
    Get shape(s) from space.

    :param space: (gym.spaces) A space whose type is one of the spaces in gym.
    :return: ((int,...) or dict) Shapes returned depending on type of space.
    '''
    if isinstance(space, Dict):
        shape = {key: get_shape(subspace)
                 for key, subspace in space.spaces.items()}
    elif isinstance(space, Box):
        shape = space.shape
    else:
        raise RuntimeError('Space type is not currently handled')
    return shape

# Tensorflow functions and classes


PlaceHolders = namedtuple('PlaceHolders', ['observations', 'actions',
                                           'rewards', 'observations_next',
                                           'dones'])


def create_default(observation_spaces, action_spaces):
    '''
    Create default placeholders for the spaces passed.

    :param observation_spaces: (gym.spaces.Dict) A Dict space.
    :param action_spaces: (gym.spaces.Dict) A Dict space.
    :return: (PlaceHolders) A namedtuple.
    '''
    observations = convert_spaces_to_placeholders(observation_spaces)
    actions = convert_spaces_to_placeholders(action_spaces)
    rewards = {name: tf.placeholder(tf.float32, shape=(None, 1))
               for name in observations}
    observations_n = convert_spaces_to_placeholders(observation_spaces)
    dones = {name: tf.placeholder(tf.float32, shape=(None, 1))
             for name in observations}
    return PlaceHolders(observations, actions, rewards, observations_n, dones)


class TfFunction:
    '''A class for running tensorflow operations as python functions.'''

    def __init__(self, session, inputs=None, outputs=None, updates=None):
        '''
        Create a TfFunction.

        :param inputs: (dict) A map of strings to placeholders.
        :param outputs: (dict) A map of strings to tensorflow operations.
        :param updates: (list) A list of tensorflow operations. Will
                               not return anything produced by operations.
        :param session: (tensorflow.Session) A session to run the function on.
        '''
        inputs = {} if inputs is None else inputs
        updates = [] if updates is None else updates
        with session.graph.as_default():
            with tf.control_dependencies(updates):
                if outputs is None:
                    outputs = {'no_op': tf.no_op()}
                else:
                    outputs = {key: tf.identity(op)
                               for key, op in outputs.items()}

        self.session = session
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, kwargs_or_none=None, **kwargs):
        '''
        Call the tensorflow operations.

        :param kwargs_or_none: (dict or None) If dict then should map string
                                              to data to feed to placeholders
                                              where the string is the key
                                              to the placeholder in inputs.
                                              Else if None then use the kwargs
                                              passed into the python function.
        :param **kwargs: Used if kwargs_or_none is None.
        :return: (dict) A dictionary containing the results of the operations.
        '''
        kwargs = kwargs if kwargs_or_none is None else kwargs_or_none
        feed_dict = {placeholder: kwarg for _, (placeholder, kwarg) in
                     zip_map(self.inputs, kwargs)}
        return self.session.run(self.outputs, feed_dict=feed_dict)
