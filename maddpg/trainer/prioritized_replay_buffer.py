"""
The DQN improvement: Prioritized Experience Replay
(based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

from math import log2, ceil

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree:
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)  # for all priority values
        # self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        size = 2 ** (ceil(log2(capacity)) + 1) - 1
        self.parent_nodes = 2 ** ceil(log2(capacity)) - 1
        self.tree = np.zeros(size, dtype=np.float64)
        # [--------------Parent nodes-------------]
        #             size: capacity - 1
        # [-------leaves to recode priority-------]
        #             size: capacity
        self.dirty = []
        self.values = []
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.data_pointer = 0

    def add(self, p, data, update=False):
        tree_idx = self.data_pointer + self.parent_nodes - 1
        self.data[self.data_pointer] = data  # update data_frame
        if update:
            self.update(tree_idx, p)  # update tree_frame
        else:
            self.dirty.append((tree_idx, p))

        self.data_pointer += 1
        # replace when exceed the capacity
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update_all(self):
        if self.dirty:
            new_dirty, new_values = list(zip(*sorted(self.dirty,
                                                     key=lambda x: x[0])))
            new_dirty = list(new_dirty)
            new_values = list(new_values)
            dirty = []
            values = []
            while new_dirty:
                idx = new_dirty.pop()
                value = new_values.pop()
                while new_dirty and new_dirty[-1] == idx:
                    value = new_values.pop()
                    new_dirty.pop()
                dirty.append(idx)
                values.append(value)
            self.dirty = []
            new_dirty = []
            new_values = []
            for idx, value in zip(dirty, values):
                new_values.append(value - self.tree[idx])
                self.tree[idx] = value
                new_dirty.append((idx - 1) // 2)
            while new_dirty:
                dirty = []
                values = []
                while new_dirty:
                    idx = new_dirty.pop()
                    value = new_values.pop()
                    while new_dirty and new_dirty[-1] == idx:
                        value += new_values.pop()
                        new_dirty.pop()
                    dirty.append(idx)
                    values.append(value)
                new_dirty = []
                new_values = values
                for idx, value in zip(dirty, values):
                    self.tree[idx] += value
                    if idx != 0:
                        new_dirty.append((idx - 1) // 2)

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        self.update_all()
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.parent_nodes + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class PrioritizedReplayMemory:  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(int(capacity))

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        max_p = np.max(self.tree.tree[-self.tree.parent_nodes:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, data)   # set the max p for new p

    def sample(self, n):
        #b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty(
        #    (n, self.tree.data[0].size)), np.empty((n, 1))
        b_idx, b_memory, ISWeights = [], [], []
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min(
            [1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # for later calculate ISweight
        min_prob = np.min(
            self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            #ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            ISWeights.append(np.power(prob/min_prob, -self.beta))
            #b_idx[i], b_memory[i, :] = idx, data
            b_idx.append(idx)
            b_memory.append(data)
        b_memory = list(zip(*b_memory))
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
