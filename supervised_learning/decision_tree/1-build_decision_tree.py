#!/usr/bin/env python3
"""
Module: 1-build_decision_tree
"""

import numpy as np


class Node:
    """
    Class: Node
    """
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Function: max_depth_below
        """
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Function: count_nodes_below
        """
        num_nodes = 0

        # if counting all nodes and current node is not a leaf
        if not only_leaves and not self.is_leaf:
            # then iclude it, too
            num_nodes += 1

        num_nodes += self.right_child
        .count_nodes_below(only_leaves=only_leaves)
        num_nodes += self.left_child.count_nodes_below(only_leaves=only_leaves)

        return num_nodes


class Leaf(Node):
    """
    Class: Leaf
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Function: max_depth_below
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Function: count_nodes_below
        """
        return 1


class Decision_Tree():
    """
    Class: Decision_Tree
    """
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
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
        """
        Function: depth
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Function: count_nodes
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
