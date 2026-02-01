#!/usr/bin/env python3
"""Line up"""

def add_arrays(arr1, arr2):
    """Line up"""
    if len(arr1) != len(arr2):
        return None

    new_matrix = [arr1[i]+arr2[i] for i in range(len(arr1))]
    return new_matrix
