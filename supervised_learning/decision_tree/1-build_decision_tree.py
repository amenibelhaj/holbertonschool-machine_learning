#!/usr/bin/env python3
"""
Module for building a Decision Tree with counting capabilities
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes the node
        """
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
        Recursively calculates the maximum depth of the tree
        """
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively counts the nodes below this node
        """
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)

        if only_leaves:
            # If counting only leaves, this node (internal) adds 0 to the sum
            return left + right

        # If counting all nodes, this node adds 1 to the sum of its children
        return 1 + left + right


class Leaf(Node):
    """
    Represents a leaf node in a decision tree
    """
    def __init__(self, value, depth=None):
        """
        Initializes the leaf
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf itself
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 because a leaf always counts as one unit
        """
        return 1


class Decision_Tree():
    """
    Represents a decision tree structure
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes the decision tree
        """
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
        Calculates the maximum depth of the entire tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the nodes/leaves in the tree starting from the root
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
