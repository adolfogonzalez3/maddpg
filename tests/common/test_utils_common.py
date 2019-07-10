'''Module for testing functions in utils_common.'''

import pytest
import maddpg.common.utils_common as utils


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
