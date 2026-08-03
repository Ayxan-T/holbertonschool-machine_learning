#!/usr/bin/env python3
"""
Module: 8-build_decision_tree
"""

import numpy as np


class Node:
    """Simple tree node used to build a decision tree."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """Initialize a node with optional split information."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        """Return a readable text view of the subtree."""
        self_type = 'root' if self.is_root else '-> node'
        tree = "{} [feature={}, threshold={}]\n" \
            .format(self_type, self.feature, self.threshold)

        left_child_tree = str(self.left_child)
        right_child_tree = str(self.right_child)

        tree += self.left_child_add_prefix(left_child_tree)
        tree += self.right_child_add_prefix(right_child_tree)

        return tree

    def max_depth_below(self):
        """Return the maximum depth below this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count the nodes below this one."""
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
        """Prefix each line of text for the left subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Prefix each line of text for the right subtree."""
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
        """Gather all leaf nodes below this node."""
        leaves = []

        left_leaves = self.left_child.get_leaves_below()
        for leaf in left_leaves:
            leaves.append(leaf)

        right_leaves = self.right_child.get_leaves_below()
        for leaf in right_leaves:
            leaves.append(leaf)

        return leaves

    def update_bounds_below(self):
        """Propagate the split bounds through the subtree."""
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
        """Create an indicator function for the node's region."""
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
        """Return the prediction for a single sample."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Leaf node that stores a final prediction value."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return a short text description of the leaf."""
        return (f"-> leaf [value={self.value}]")

    def max_depth_below(self):
        """Return the leaf depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count this leaf as one node."""
        return 1

    def get_leaves_below(self):
        """Return this leaf as the only leaf below it."""
        return [self]

    def update_bounds_below(self):
        """Do nothing because a leaf has no children."""
        pass

    def pred(self, x):
        """Return the stored leaf prediction."""
        return self.value


class Decision_Tree():
    """Fit a decision tree and use it for prediction."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Initialize a decision tree with basic training settings."""
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
        
        # predict all individuals in A
        self.predict = lambda A: [self.pred(individual) for individual in A]

    def pred(self, x):
        """Predict the label for a single sample."""
        return self.root.pred(x)
    
    def fit(self,explanatory, target, verbose=0):
        """Train the decision tree on the provided data.
        
                Args:
                    explanatory: A 2D array of feature values for the training data.
                    target: A 1D array of labels corresponding to the training data.
                    verbose: If set to 1, print training summary information.
        
                Returns:
                    None
        """
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
        """Return the minimum and maximum values of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Choose a random feature and threshold for splitting."""
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
        """Recursively fit a node and its children."""
        node.feature, node.threshold = self.split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        left_population  = \
            node.sub_population & (feature_values > node.threshold)
        right_population = \
            node.sub_population & (feature_values <= node.threshold)

        # Is left node a leaf ?
        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            self.max_depth == node.depth + 1 or
            np.all(
                self.target[left_population] == \
                    self.target[left_population][0]
            )
        )

        if is_left_leaf :
                node.left_child = self.get_leaf_child(node,left_population)                                                         
        else :
                node.left_child = self.get_node_child(node,left_population)
                self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            self.max_depth == node.depth + 1 or
            np.all(
                self.target[right_population] == \
                    self.target[right_population][0]
            )
        )

        if is_right_leaf :
                node.right_child = self.get_leaf_child(node,right_population)
        else :
                node.right_child = self.get_node_child(node,right_population)
                self.fit_node(node.right_child)    

    def get_leaf_child(self, node, sub_population):
        """Create a leaf node from the majority label in a subset."""
        values = self.target[sub_population]
        unique, counts = np.unique(values, return_counts=True)
        value = unique[np.argmax(counts)] # mode value
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node for a subtree."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Return the classification accuracy on a dataset."""
        return np.sum(np.equal(
            self.predict(test_explanatory),
            test_target
        )) / test_target.size
    
    def possible_thresholds(self, node, feature):
        """List possible midpoints between unique feature values."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:]+values[:-1])/2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Evaluate the best threshold for one feature using Gini impurity."""
        # Compute a numpy array of booleans Left_F of shape (n,t,c) where
            #    -> n is the number of individuals in the sub_population
            #                                       corresponding to node
            #    -> t is the number of possible thresholds
            #    -> c is the number of classes represented in node
            # such that Left_F[ i , j , k] is true iff 
            #    -> the i-th individual in node is of class k 
            #    -> the value of the chosen feature on the i-th individual 
            #                   is greater than the t-th possible threshold
            # then by squaring and summing along 2 of the axes of
            # Left_F[ i , j , k], you can get the Gini impurities of the
            # putative left childs as a 1D numpy array of size t 
            #
            # Then do the same with the right child
            # Then compute the average sum of these Gini impurities
            #
            # Then  return the threshold and the Gini average  for which the
            # Gini average is the smallest
        values = self.explanatory[:, feature][node.sub_population] # (n, d)
        targets = self.target[node.sub_population]     # (n,)
        thresholds = self.possible_thresholds(node, feature)       # (t,)

        # 1. Calculating left leaf Gini impurities (> threshold)
        left_mask = values[:, None] > thresholds[None, :] # (n, t)

        classes, class_ids = np.unique(targets, return_inverse=True)
        right_mask = np.eye(len(classes))[class_ids]  # (n, c)

        Left_F = np.logical_and(
            left_mask[:, :, None], right_mask[:, None, :]) # (n, t, c)

        # Calculate total number of left leaf points for every threshold
        left_totals = np.sum(Left_F, axis=(0, 2), keepdims=False) # (t,)

        # Calculate sum of squared probabilities
        squared_probs = np.square(
            np.sum(Left_F, axis=0, keepdims=False)) # (t, c)
        sumof_sqrd_probs = np.sum(squared_probs, axis=1) # (t,)

        Gini_Left = 1 - sumof_sqrd_probs / left_totals ** 2

        # 2. Calculating rigth leaf Gini impurities (<= threshold)
        left_mask = values[:, None] <= thresholds[None, :] # (n, t)

        Right_F = np.logical_and(
            left_mask[:, :, None], right_mask[:, None, :]) #(n, t, c)

        right_totals = np.sum(Right_F, axis=(0, 2), keepdims=False) # (t,)

        squared_probs = np.square(
            np.sum(Right_F, axis=0, keepdims=False)) # (t, c)
        sumof_sqrd_probs = np.sum(squared_probs, axis=1) # (t,)

        Gini_Right = 1 - sumof_sqrd_probs / right_totals ** 2

        # 3. Calculate Gini average
        Gini_ave = (
            left_totals * Gini_Left +
            right_totals * Gini_Right
        ) / (left_totals + right_totals)

        best_index = np.argmin(Gini_ave)
        return thresholds[best_index], Gini_ave[best_index]


    def Gini_split_criterion(self, node):
        """Choose the best feature and threshold using Gini impurity."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i) for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
