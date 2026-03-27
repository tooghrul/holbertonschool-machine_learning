#!/usr/bin/env python3
"""
Module: decision_tree

This module defines a simple Decision Tree structure with three classes:
- Node: represents an internal decision node
- Leaf: represents a terminal node (leaf)
- Decision_Tree: wrapper class for the tree

It also provides functionality to retrieve all leaves from the tree.
"""
import numpy as np

class Node:
    """
    Represents an internal node in a decision tree.

    Attributes:
        feature (int | None): Feature index used for splitting.
        threshold (float | None): Threshold value for the split.
        left_child (Node | Leaf | None): Left subtree.
        right_child (Node | Leaf | None): Right subtree.
        depth (int): Depth of the node in the tree.
        is_root (bool): Whether this node is the root.
        is_leaf (bool): Indicates if the node is a leaf.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 depth=0, is_root=False):
        """
        Initializes a Node.

        Args:
            feature (int, optional): Feature index.
            threshold (float, optional): Threshold for decision.
            left_child (Node or Leaf, optional): Left child node.
            right_child (Node or Leaf, optional): Right child node.
            depth (int, optional): Depth of node in tree.
            is_root (bool, optional): Marks node as root.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def get_leaves_below(self):
        """
        Recursively collects all leaf nodes in the subtree
        rooted at this node.

        Returns:
            list[Leaf]: List of all leaf nodes under this node.
        """
        leaves = []

        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())

        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves


class Leaf(Node):
    """
    Represents a leaf (terminal node) in a decision tree.

    Attributes:
        value (any): The value stored in the leaf.
    """

    def __init__(self, value, depth=0):
        """
        Initializes a Leaf.

        Args:
            value (any): Value stored in the leaf.
            depth (int, optional): Depth of leaf in tree.
        """
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True

    def get_leaves_below(self):
        """
        Returns itself as it is a leaf.

        Returns:
            list[Leaf]: A list containing only this leaf.
        """
        return [self]

    def __str__(self):
        """
        String representation of the leaf.

        Returns:
            str: Formatted string showing leaf value.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    Represents a decision tree.

    Attributes:
        root (Node | Leaf): Root of the tree.
    """

    def __init__(self, root=None):
        """
        Initializes a Decision Tree.

        Args:
            root (Node or Leaf, optional): Root node of the tree.
        """
        self.root = root

    def get_leaves(self):
        """
        Retrieves all leaves in the tree.

        Returns:
            list[Leaf]: List of all leaf nodes in the tree.
        """
        return self.root.get_leaves_below()
