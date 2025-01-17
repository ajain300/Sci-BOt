from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Union, Dict
import ast

# Active Learning library imports
from .utils.data_utils import preprocess_monomer_list
import warnings


class RequirementChecker(ABC):
    @abstractmethod
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        pass

    def _get_columns(self, columns: Union[str, List[str], Dict[str, str]]) -> List[str]:
        if isinstance(columns, str):
            return [columns]
        elif isinstance(columns, list):
            return columns
        elif isinstance(columns, dict):
            columns_list = []
            for key, value in columns.items():
                if isinstance(value, dict):
                    columns_list.append(value['name'])
                else:
                    columns_list.append(value)
            return columns_list
        else:
            raise ValueError(f"Unsupported column type: {type(columns)}")

class UniqueMonomerChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        if isinstance(columns, dict):
            monomer_col, ratio_col = columns['parts'], columns['ratios']
        else:
            monomer_col, ratio_col = columns

        data[monomer_col] = data[monomer_col].apply(preprocess_monomer_list)
        monomers = data[monomer_col].explode().unique()
        if len(monomers) != len(set(monomers)):
            raise ValueError("Monomers are not unique")

class NoMissingValuesChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        columns = self._get_columns(columns)
        if isinstance(columns, dict):
            columns = list(columns.values())
        if data[columns].isnull().any().any():
            raise ValueError(f"Missing values found in columns: {columns}")

class RatiosSumToNumberChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str]], params: dict = None):
        sum_number = params
        columns = self._get_columns(columns)
        if not np.allclose(data[columns].apply(sum, axis = 1), sum_number):
            raise ValueError(f"1 or more ratios do not sum to {sum_number}. The summed ratios are: {data[columns].apply(sum, axis = 1)}")

class NonNegativeRatiosChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        columns = self._get_columns(columns)
        if (data[columns].apply(lambda x: any(val < 0 for val in x))).any():
            raise ValueError("Negative ratios found")

class CategoricalChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        columns = self._get_columns(columns)
        if not all(data[col].dtype == 'object' for col in columns):
            raise ValueError(f"Columns {columns} should be categorical")

class NumericChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        columns = self._get_columns(columns)
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in columns):
            raise ValueError(f"Columns {columns} should be numeric")

class RangeChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        min_val, max_val = params[0], params[1]
        columns = self._get_columns(columns)
        for col in columns:
            if not ((data[col].isnull()) | ((data[col] >= min_val) & (data[col] <= max_val))).all():
                raise ValueError(f"Values in {col} should be between {min_val} and {max_val}, or null")

class StdNonNegativeChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        if isinstance(columns, dict):
            std_col = columns['std']
        else:
            std_col = columns[1]
        if (data[std_col] < 0).any():
            raise ValueError(f"Standard deviation column {std_col} contains negative values")

class StdSmallerThanValueChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        if isinstance(columns, dict):
            value_col, std_col = columns['value'], columns['std']
        else:
            value_col, std_col = columns
        if (data[std_col] > data[value_col]).any():
            warnings.warn(f"Some standard deviations in {std_col} are larger than their corresponding values in {value_col}")

class AllowedValuesChecker(RequirementChecker):
    def check(self, data: pd.DataFrame, columns: Union[str, List[str], Dict[str, str]], params: dict = None):
        allowed_values = params
        columns = self._get_columns(columns)
        if not all(data[columns].isin(allowed_values).all()):
            raise ValueError(f"Values in {columns} should be one of {allowed_values}")


class RequirementCheckerFactory:
    @staticmethod
    def get_checker(checker_name: str) -> RequirementChecker:
        checkers = {
            'unique_monomers': UniqueMonomerChecker(),
            'no_missing_values': NoMissingValuesChecker(),
            'ratios_sum_to_number': RatiosSumToNumberChecker(),
            'non_negative_ratios': NonNegativeRatiosChecker(),
            'categorical': CategoricalChecker(),
            'numeric': NumericChecker(),
            'range': RangeChecker(),
            'std_non_negative': StdNonNegativeChecker(),
            'std_smaller_than_value': StdSmallerThanValueChecker(),
            'allowed_values': AllowedValuesChecker(),
        }
        return checkers.get(checker_name)