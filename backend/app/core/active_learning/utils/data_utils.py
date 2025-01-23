# Helper funcitons to help with data formatting/processing tasks
from typing import List
import pandas as pd
import numpy as np

# Helper function to transform monomer list into list of strings
def preprocess_monomer_list(monomer_str):
    # Remove the brackets and split by comma
    monomers = monomer_str.strip('[]').split(',')
    # Remove extra spaces and ensure no quotes around elements
    monomers = [monomer.strip() for monomer in monomers]
    return monomers


# Helper function to transform the monomers into columns
def create_monomer_dict(row):
    monomer_dict = dict(zip(row['Monomers'], row['Ratio']))
    for monomer in self.monomer_list:
        if monomer not in monomer_dict:
            monomer_dict[monomer] = 0.
    return pd.Series(monomer_dict)

def one_hot_encode(number: int, num_classes: int) -> np.ndarray:
    """
    Converts a number into a one-hot encoding.

    :param number: The number to be encoded.
    :param num_classes: The total number of classes.
    :return: The one-hot encoding of the number.
    :rtype: List[int]
    """
    encoding = [0] * num_classes
    encoding[number] = 1
    return np.array(encoding)

def flatten(nested_list):
    """Flatten a nested list."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list