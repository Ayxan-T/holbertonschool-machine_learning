#!/usr/bin/env python3
"""Module: 10-isolation_tree"""

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf
import numpy as np


class Isolation_Random_Tree:
    """Isolation random tree for anomaly detection."""

    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return a text representation of the tree."""
        return self.root.__str__()

    def depth(self):
        """Return the tree depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Return the leaf nodes of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the decision-region bounds for each node."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Prepare the prediction function after fitting the tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

    def np_extrema(self, arr):
        """Return the minimum and maximum values of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Choose a random feature and threshold for splitting."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf node for a terminal subtree."""
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node for a subtree."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fit a node and its children."""
        node.feature, node.threshold = self.random_split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        left_population = node.sub_population & (
            feature_values > node.threshold
        )
        right_population = node.sub_population & (
            feature_values <= node.threshold
        )

        # Is left node a leaf ?
        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            self.max_depth == node.depth + 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            self.max_depth == node.depth + 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fit the isolation tree to the explanatory data."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")