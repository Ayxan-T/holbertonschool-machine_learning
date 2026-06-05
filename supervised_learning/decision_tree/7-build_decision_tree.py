#!/usr/bin/env python3
"""
Module: 7-build_decision_tree
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

    def __str__(self):
        self_type = 'root' if self.is_root else '-> node'
        tree = "{} [feature={}, threshold={}]\n" \
            .format(self_type, self.feature, self.threshold)

        left_child_tree = str(self.left_child)
        right_child_tree = str(self.right_child)

        tree += self.left_child_add_prefix(left_child_tree)
        tree += self.right_child_add_prefix(right_child_tree)

        return tree

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

        num_nodes += self.right_child \
            .count_nodes_below(only_leaves=only_leaves)
        num_nodes += self.left_child.count_nodes_below(only_leaves=only_leaves)

        return num_nodes

    def left_child_add_prefix(self, text):
        """
        Function: left_child_add_prefix
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        Function: right_child_add_prefix
        """
        lines = text.split("\n")
        if len(lines) == 1:
            new_text = "    +--" + lines[0]
        else:
            new_text = "    +--" + lines[0] + "\n"

        if len(lines) == 2:
            new_text += "       "+lines[1]

            return new_text

        if len(lines) > 2:
            for x in lines[1:-1]:
                new_text += ("       " + x) + "\n"

            new_text += ("       " + lines[-1])

        return (new_text)

    def get_leaves_below(self):
        """
        Function: get_leaves_below
        """
        leaves = []

        left_leaves = self.left_child.get_leaves_below()
        for leaf in left_leaves:
            leaves.append(leaf)

        right_leaves = self.right_child.get_leaves_below()
        for leaf in right_leaves:
            leaves.append(leaf)

        return leaves

    def update_bounds_below(self):
        """
        Function: update_bounds_below
        """
        if self.is_root:
            self.upper = {}
            self.lower = {}

        for child in [self.left_child, self.right_child]:
            # 1. Copy the current node's bounds to the child
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            # 2. Update the specific bound changed by this split
            if child == self.left_child:
                # Left child takes values > threshold
                child.lower[self.feature] = self.threshold
            else:
                # Right child takes values <= threshold
                child.upper[self.feature] = self.threshold

        # Recursively call the method for children to propagate further down
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Function: update_indicator
        """
        def is_large_enough(x):
            return np.logical_and.reduce([
                x[:, key] > self.lower[key]
                for key in self.lower
            ])

        def is_small_enough(x):
            return np.logical_and.reduce([
                x[:, key] <= self.upper[key]
                for key in self.upper
            ])

        # True if all values are within (lower, upper)
        self.indicator = lambda x: np.logical_and(
            is_large_enough(x),
            is_small_enough(x)
        )

    def pred(self, x):
        """
        Function: pred
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Class: Leaf
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return (f"-> leaf [value={self.value}]")

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

    def get_leaves_below(self):
        """
        Function: get_leaves_below
        """
        return [self]

    def update_bounds_below(self):
        """
        Function: update_bounds_below
        """
        pass

    def pred(self, x):
        """
        Function: pred
        """
        return self.value


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

    def __str__(self):
        return self.root.__str__()

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

    def get_leaves(self):
        """
        Funciton: get_leaves
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Function: update_bounds
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Function: update_predict
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        
        # predict all individuals in A
        self.predict = lambda A: [self.pred(individual) for individual in A]

    def pred(self, x):
        """
        Function: pred
        """
        return self.root.pred(x)
    
    def fit(self,explanatory, target, verbose=0):
        if self.split_criterion == "random": 
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target,dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose==1:
            print(f"""  Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(only_leaves=True)}
- Accuracy on training data : {self.accuracy(self.explanatory,self.target)}""")
    
    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0 :
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:,feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold
    
    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)

        left_population  =
        right_population =

        # Is left node a leaf ?
        is_left_leaf =

        if is_left_leaf :
                node.left_child = self.get_leaf_child(node,left_population)                                                         
        else :
                node.left_child = self.get_node_child(node,left_population)
                self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf =

        if is_right_leaf :
                node.right_child = self.get_leaf_child(node,right_population)
        else :
                node.right_child = self.get_node_child(node,right_population)
                self.fit_node(node.right_child)    

    def get_leaf_child(self, node, sub_population):
        value =
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(
            self.predict(test_explanatory),
            test_target
        )) / test_target.size
