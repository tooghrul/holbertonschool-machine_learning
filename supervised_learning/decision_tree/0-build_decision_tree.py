#!/usr/bin/env python3
""" Implementing Decision Tree """
import numpy as np


class Node:
    """ The node of the tree """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """ Initialize Node """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Function for finding depth of the tree """
        if self.is_leaf:
            return self.depth
        return max(
            self.depth,
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )


class Leaf(Node):
    """ The leaf of the tree """
    def __init__(self, value, depth=None):
        """ Initialize leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Depth of the tree """
        return self.depth


class Decision_Tree():
    """ Class for implementing Decision Tree """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ Initialize Tree """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ Function for finding depth of the tree """
        return self.root.max_depth_below()
