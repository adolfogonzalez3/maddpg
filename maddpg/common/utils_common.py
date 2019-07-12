'''A module that contains utilites commonly used throughout the library.'''

from collections import ChainMap, defaultdict

import tensorflow as tf
from gym.spaces import Box, Discrete


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


def flatten_map(dictionary):
    '''
    Flatten a nested dict by prefixing a subdict's key with its parent.

    '''
    new_map = {}
    for parent, child_dict in dictionary.items():
        for child, value in child_dict.items():
            new_map['{}_{}'.format(parent, child)] = value
    return new_map


def unflatten_map(dictionary):
    '''
    Reverse prior flattening of a dictionary.
    '''
    new_map = defaultdict(dict)
    for key, value in dictionary.items():
        parent, child = key.split('_')
        new_map[parent][child] = value
    return dict(new_map)


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
