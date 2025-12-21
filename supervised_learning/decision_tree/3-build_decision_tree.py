#!/ --- 3-build_decision_tree.py --- /

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def get_leaves_below(self):
        """Recursively gathers all leaves sitting below this node"""
        return self.left_child.get_leaves_below() + self.right_child.get_leaves_below()


class Leaf:
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def get_leaves_below(self):
        """A leaf is the end of the line, so it just returns itself in a list"""
        return [self]

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="gini", root=None):
        self.root = root
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        np.random.seed(seed)

    def get_leaves(self):
        """Starts the search for leaves from the very top of the tree"""
        return self.root.get_leaves_below()
