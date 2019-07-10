'''A module that contains utilites commonly used throughout the library.'''


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
