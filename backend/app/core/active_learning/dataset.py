import pandas as pd
from enum import Enum
import ast
import numpy as np
from typing import Dict, Any, List, Union, Tuple
import yaml
import itertools
import logging

# imports from inside active learning
from .requirement_checkers import RequirementCheckerFactory, RequirementChecker
from .utils.data_utils import one_hot_encode, flatten
from .utils.constants import *
from .targets import *
from .features import *
import warnings

from ...schemas.optimization import OptimizationConfig, DataPoint, OptimizationDirection
from ...schemas.feature_schemas import *

logger = logging.getLogger(__name__)

class VariableType(Enum):
    composition = "composition" 
    discrete = "discrete"
    correlated = "correlated"
    target = "target"
    general = "general"
    
class AL_Dataset:
    """AL_Dataset is a class designed for handling datasets used in active learning processes. It provides methods for reading, processing, and normalizing data, as well as generating candidate spaces for new data points.
    Attributes:
        minimum_feature_keys (list): List of minimum required keys for features.
        minimum_target_keys (list): List of minimum required keys for targets.
        logger (logging.Logger): Logger for the class.
        config (dict): Configuration dictionary for the dataset.
        targets (dict): Dictionary to store target features.
        dataset_stats (dict): Dictionary to store dataset statistics.
        feature_columns (dict): Dictionary to store feature columns.
        features (dict): Dictionary to store features.
        drop_X_columns (list): List of feature columns to be dropped in model input due to data constraints.
    Methods:
        __init__(data_config: Union[str, Dict], **kwargs):
            Initializes the AL_Dataset object with the given configuration.
        __repr__():
            Returns a string representation of the AL_Dataset object.
        read_dataset(path: str):
            Reads the dataset from the given path.
        set_dataset(dataset: pd.DataFrame):
            Sets the dataset to the given DataFrame.
        X_columns():
            Returns a list of columns used as features (excluding target features).
        X_columns_pred():
            Returns a list of columns used as features for prediction (excluding target features and dropped columns).
        process_data() -> pd.DataFrame:
            Processes the data based on the configuration provided, handling different types of features and their requirements.
        _process_feature_data(data: pd.DataFrame, feature_type: VariableType, columns: Any, name: str) -> pd.DataFrame:
            Processes feature data based on the feature type and columns.
        _process_targets(data: pd.DataFrame):
            Processes target data and returns the modified DataFrame.
        _process_requirements(data: pd.DataFrame, feature_type: VariableType, columns, requirements: List):
            Processes requirements for a given feature or column.
        _normalize_properties(data: pd.DataFrame):
            Normalizes properties in the DataFrame.
        get_training_format(**kwargs):
            Returns the training format of the dataset, including scaled matrices used in training.
        get_inference_format(data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
            Returns the inference format of the dataset.
        normalize(data: np.ndarray, variable: str, data_std=None):
            Normalizes the given data based on the variable's statistics.
        unnormalize(data, variable, data_std=None):
            Unnormalizes the given data based on the variable's statistics.
        generate_candidate_space(**kwargs):
            Generates a candidate space for new data points based on the configuration.
        get_dataset_stats():
            Returns the dataset statistics.
        make_rec_readable(candidates_df):
            Converts one-hot encoded columns in the candidates DataFrame to readable format."""
    minimum_feature_keys = ['name', 'type']
    minimum_target_keys = ['name', 'direction', 'scaling', 'unit']

    def __init__(self, data_config : OptimizationConfig, data : List[DataPoint]):
        self.config = data_config
        
        for feature_config in self.config.features:
            logger.debug("Printing feature configs")
            logger.debug(f"Feature config: {feature_config}")
        self._set_dataset(data)
        
        # initialize properties information as empty
        self.targets = {}
        self.variable_info = {}
        self.dataset_stats = {}
        self.feature_columns = {} # dict of feature_name: columns for the feature in the dataset
        self.features = {}
        
        # Store X_columns that need to be dropped
        self.drop_X_columns = []

    def __repr__(self):
        return 'ActiveLearningDataset'

    def _set_dataset(self, dataset: List[DataPoint]) -> None:
        """
        Convert a list of DataPoint objects to a pandas DataFrame.
        
        Args:
            dataset: List of DataPoint objects containing parameters and objective values
        """
        print("SETTING DATASET")
        if not dataset:
            # Create empty DataFrame with correct columns
            parameter_cols = list(self.config.features.keys())
            objective_cols = [obj.name for obj in self.config.objectives]
            self.wide_dataset = pd.DataFrame(columns=parameter_cols + objective_cols)
            return

        # Convert list of DataPoints to list of dictionaries
        rows = []
        for point in dataset:
            row = point.parameters.copy()  # Start with parameters
            row.update(point.objective_values)  # Add objective values
            rows.append(row)
        
        # Create DataFrame from list of dictionaries
        self.wide_dataset = pd.DataFrame(rows)
        
        # Ensure all expected columns are present
        expected_parameter_cols = extract_parameter_columns(self.config.features)
        logger.debug(f"expected parameter cols in df {expected_parameter_cols}")
        
        expected_objective_cols = [obj.name for obj in self.config.objectives]
        expected_cols = expected_parameter_cols + expected_objective_cols
        logger.debug(f"expected cols in df {expected_cols}")
        missing_cols = set(expected_cols) - set(self.wide_dataset.columns)
        for col in missing_cols:
            self.wide_dataset[col] = float('nan')  # Use NaN for missing values
        logger.debug("wide_dataset" + self.wide_dataset.to_string())

        # Reorder columns to ensure consistent format
        self.wide_dataset = self.wide_dataset[expected_cols]
    @property
    def X_columns(self):
        return [col for feat in self.features.values() for col in feat.X_columns if not isinstance(feat, Target)]

    @property
    def X_columns_pred(self):
        return [col for feat in self.features.values() for col in feat.X_columns if not isinstance(feat, Target) and col not in self.drop_X_columns]

    def process_data(self) -> pd.DataFrame:
        """
        Process data based on the configuration provided, handling different types of
        features and their requirements.
        
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        logger.debug("Starting process_data")
        data = self.wide_dataset
        
        logger.debug(f"Data before processing {data}")
        
        # PROCESS FEATURES
        for feature_config in self.config.features:
            logger.debug(f"Processing feature: {feature_config.name}")
            logger.debug(f"Feature config: {feature_config}")
            # Create appropriate feature object using factory
            feature = create_feature(feature_config)
            feature.process()
            
            # Store feature and its columns
            self.features[feature.name] = feature
            self.X_columns.extend(feature.X_columns)
            
            # Process the feature data
            data = self._process_feature_data(data, feature)

        logger.debug(f"data after feature process {data.to_string()}")
        # PROCESS OBJECTIVES
        for objective in self.config.objectives:
            logger.debug(f"Processing objective: {objective.name}")
            
            # Check if all required keys are present in ObjectiveConfig
            required_keys = {'name', 'direction', 'weight'}
            missing_keys = required_keys - set(objective.dict().keys())
            
            if missing_keys:
                raise ValueError(f"Missing required values for objective {objective.name}: {missing_keys}")

            # Set objective info
            columns = {
                'value': objective.name,
                'std': f"{objective.name}_std"
            }
            
            properties = {
                'type': objective.direction.value,  # Using OptimizationDirection enum value
                'weight': objective.weight,
                'value_col': columns['value'],
                'std_col': columns['std'],
                'scaling': objective.scaling,
                'unit': objective.unit if objective.unit else None,
            }
            
            # Add target-specific properties
            if objective.direction == OptimizationDirection.TARGET and objective.target_value is not None:
                properties['target_value'] = objective.target_value
            elif objective.direction == OptimizationDirection.RANGE:
                if objective.range_min is not None:
                    properties['range_min'] = objective.range_min
                if objective.range_max is not None:
                    properties['range_max'] = objective.range_max
            
            # For this variable, if std column isn't in the dataset then assume 10%
            if columns['std'] not in data.columns:
                data[columns['std']] = 0.1 * data[columns['value']].mean()
            
            # Create target config
            target_config = {
                'name': objective.name,
                'weight': objective.weight,
                'scaling': objective.scaling,
                'unit': objective.unit
            }
            
            # Map optimization direction to target type and direction
            if objective.direction in [OptimizationDirection.MAXIMIZE, OptimizationDirection.MINIMIZE]:
                target_config['type'] = 'extreme'
                target_config['direction'] = 'MAX' if objective.direction == OptimizationDirection.MAXIMIZE else 'MIN'
            elif objective.direction == OptimizationDirection.TARGET:
                target_config['type'] = 'target'
                target_config['direction'] = 'TARGET'
                target_config['target_value'] = objective.target_value
            elif objective.direction == OptimizationDirection.RANGE:
                target_config['type'] = 'range'
                target_config['direction'] = 'RANGE'
                target_config['range_min'] = objective.range_min
                target_config['range_max'] = objective.range_max

            # Create target object using factory
            target_obj = create_target(
                config=target_config,
                X_columns=list(columns.values()),
                variable_info={'scaling': objective.scaling},
                properties={
                    'type': objective.direction.value,
                    'weight': objective.weight,
                    'value_col': columns['value'],
                    'std_col': columns['std'],
                    'scaling': objective.scaling,
                    'unit': objective.unit
                }
            )
            target_obj.process()
            self.targets[objective.name] = target_obj

        data = self._process_targets(data)
        data = self._normalize_properties(data)
        
        self.input_data = data
        logger.debug("Finished process_data")
        return data

    def _process_feature_data(self, data: pd.DataFrame, feature: FeatureBase) -> pd.DataFrame:
        """Process feature data based on the feature type."""
        logger.debug(f"Processing feature data for {feature.name}")
        logger.debug(f"is composition feature: {isinstance(feature, CompositionFeature)}")
        logger.debug(f"is discrete feature: {isinstance(feature, DiscreteFeature)}")
        logger.debug(f"is continuous feature: {isinstance(feature, ContinuousFeature)}")
        logger.debug(f"is feature base: {isinstance(feature, Feature)}")
        logger.debug(f"feature type: {type(feature)}")
        logger.debug(f"feature name: {feature.name}")
        if isinstance(feature, CompositionFeature):
            logger.debug(f"{feature.name} is composition feature")
            # Process composition feature
            for part in feature.parts:
                if data[part].nunique() == 1:
                    data = data.drop(columns=part)
                    self.drop_X_columns.append(part)
                    logger.warning(f"Dropping {part} as it has only one unique value")
            
        elif isinstance(feature, DiscreteFeature):
            logger.debug(f"{feature.name} is discrete feature")
            # Process discrete feature by one-hot encoding
            col = feature.name
            data = pd.get_dummies(data, columns=[col], prefix=col)
            
            logger.debug(f"data after one-hot encoding feature {feature.name}: {data.to_string()}")
            
            # Update feature columns with one-hot encoded columns
            one_hot_columns = [c for c in data.columns if c.startswith(f"{col}_")]
            feature.X_columns = one_hot_columns
            # Store one-hot encoding information
            feature.add_OH_info(one_hot_columns)
            
        elif isinstance(feature, ContinuousFeature):
            logger.debug(f"{feature.name} is continuous feature")
            # Process continuous feature
            col = feature.name
            if data[col].nunique() == 1:
                data = data.drop(columns=col)
                self.drop_X_columns.append(col)
                logger.warning(f"Dropping {col} as it has only one unique value")
        return data

    def _process_targets(self, data: pd.DataFrame):
        # Melting the DataFrame
        logger.debug("Processing targets")

        
        value_columns = [target.properties['value_col'] for target in self.targets.values()]
        std_columns = [target.properties['std_col'] for target in self.targets.values()]
        logger.debug(f"value and std cols, {value_columns} , {std_columns}")
        melted = pd.melt(data, id_vars=self.X_columns, 
                            value_vars=value_columns + std_columns, 
                            var_name='Y_variable', value_name='Value')

            # Identify whether each row is a Value or Variance
        melted['is_std'] = np.where(
            melted['Y_variable'].isin(std_columns),
            'Y_variance',
            'Y_value'
        )

        # Map standard deviation columns back to their corresponding value columns
        std_to_value = dict(zip(std_columns, value_columns))
        melted['Y_variable'] = melted['Y_variable'].replace(std_to_value)

        # Pivot the 'is_std' column to create 'Y_value' and 'Y_variance' columns
        # Instead of using pivot_table, we'll use groupby and unstack to avoid memory issues
        grouped = melted.groupby(self.X_columns + ['Y_variable', 'is_std'])['Value'].first().unstack()
        data = grouped.reset_index()
        
        # Pivoting the table to wide format to separate value and variance       
        data = data.dropna(subset = ["Y_value"]).reset_index(drop = True)

        # Renaming columns for clarity
        data.columns.name = None  # Remove the categorization
        data = data.fillna({"Y_variance" : 0})

        # Add column for property number
        data["Prop_Number"] = data["Y_variable"].apply(lambda x: value_columns.index(x))
        one_hot_encoded = pd.get_dummies(data, columns = ["Y_variable"], prefix = "Property_type", dtype = int)
        data = pd.concat([one_hot_encoded, data["Y_variable"]], axis = 1)
        one_hot_columns = [col for col in data.columns if col.startswith("Property_type")]
        # Create a dictionary to map each category to its one-hot encoded vector
        categories = data['Y_variable'].unique()
        one_hot_dict = {category: [list(data.loc[data['Y_variable'] == category, one_hot_columns].iloc[0])] for category in categories}

        # data[one_hot_columns] = data[one_hot_columns].astype(int)
        for target in self.targets.values():
            target.variable_info['one_hot_column'] = "Property_type_" + target.properties['value_col']
            target.variable_info['one_hot_encoding'] = one_hot_dict[target.properties['value_col']]

        logger.debug(f"data after target process {data.to_string()}")

        return data

    def _process_requirements(self, data: pd.DataFrame, feature_type: VariableType, columns, requirements: List):
        """
        Process requirements for a given feature or column.

        :param data: Input DataFrame
        :param feature_type: Type of the feature
        :param columns: Column or columns to check
        :param requirements: List of requirements to process
        """
        for requirement in requirements:
            if isinstance(requirement, dict):
                req_name = list(requirement.keys())[0]
                req_params = requirement[req_name]
            else:
                req_name = requirement
                req_params = None

            checker : RequirementChecker = RequirementCheckerFactory.get_checker(req_name)
            if not checker:
                raise ValueError(f"Unknown requirement checker: {req_name}")

            checker.check(data, columns, req_params)
            # if feature_type == VariableType.composition and isinstance(columns, dict):
            #     checker.check(data, columns['parts'], columns['ratios'], req_params)
            # elif feature_type == VariableType.target and isinstance(columns, dict):
            #     checker.check(data, columns['value'], columns['std'], req_params)
            # else:
            #     checker(data, columns, req_params)

    def _normalize_properties(self, data : pd.DataFrame):
        # Function to apply normalization
        def normalize_group(group):
            normalized_values, normalized_std = self.normalize(group['Y_value'].values, group['Y_variable'].values[0], data_std = group['Y_variance'].values)
            group['Y_value_normalized'] = normalized_values
            group['Y_value_std_normalized'] = normalized_std
            return group

        # Apply normalization to each group
        data = data.groupby('Y_variable').apply(normalize_group).reset_index(drop=True)

        return data

    def get_training_format(self, **kwargs):
        """
        Uses the dataset format to output scaled matrices used in training the samples.
        """
        X = self.input_data[self.X_columns].copy()
        for col in self.X_columns:
            X.loc[:, f"{col}_normalized"] = self.normalize(X[col].values, col)
            
        one_hot_columns = [col for col in self.input_data.columns if col.startswith("Property_type")]
        Y_type = np.array(self.input_data[one_hot_columns].values)
        
        if len(self.targets) == 1:
            X_train = np.array(X[[f"{col}_normalized" for col in self.X_columns if col not in self.drop_X_columns]])
        else:
            X_train = np.concatenate([np.array(X[[f"{col}_normalized" for col in self.X_columns]]), Y_type], 
                         axis = 1)

        Y = self.input_data['Y_value_normalized'].values
        
        Y_std = self.input_data['Y_value_std_normalized'].values
        return X_train, Y, Y_std

    def get_inference_format(self, data : pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        Inference format for the dataset. Assumes one property at a time.
        TODO Add support for multiple properties at once
        """
        
        X = data[self.X_columns].copy()
        for col in self.X_columns:
            X.loc[:, f"{col}_normalized"] = self.normalize(X[col].values, col)
        
        target_prediction = kwargs.get('target_property', next(iter(self.targets)))
        Y_type = self.targets[target_prediction].variable_info['one_hot_encoding'][0]
        Y_type_add = np.array([Y_type for _ in range(len(X))])

        if len(self.targets) == 1:
            X_pred = np.array(X[[f"{col}_normalized" for col in self.X_columns if col not in self.drop_X_columns]])
            return X_pred, self.X_columns_pred
        else:
            Y_type_add = np.array([Y_type for _ in range(len(X))])
            X_pred = np.concatenate([np.array(X[[f"{col}_normalized" for col in self.X_columns]]), Y_type_add], 
                         axis = 1)
            return X_pred, self.X_columns_pred + [f"Property_{target.name}" for target in self.targets.values()]
        
    def normalize(self, data: np.ndarray, variable: str, data_std = None):
        """Normalize data based on the variable's configuration and statistics.
        
        Args:
            data: Data to normalize
            variable: Variable name
            data_std: Optional standard deviation data
            
        Returns:
            Normalized data and optionally normalized standard deviation
        """
        # Get variable object
        feature_obj = None
        for prop_name, prop_obj in self.features.items():
            if variable in prop_obj.X_columns:
                feature_obj = prop_obj
                break
        
        if feature_obj is None:
            for prop_name, prop_obj in self.targets.items():
                if variable in prop_obj.X_columns:
                    feature_obj = prop_obj
                    break
                
        logger.debug(f"feature object for {variable}: {feature_obj}")

        if feature_obj is None:
            raise ValueError(f"Variable {variable} not found in features or targets")
        
        # Get scaling type - check both properties and direct attribute
        scaling = None
        if hasattr(feature_obj, 'properties') and isinstance(feature_obj.properties, dict):
            scaling = feature_obj.properties.get('scaling')
        if scaling is None and hasattr(feature_obj, 'scaling'):
            scaling = feature_obj.scaling
        if scaling is None:
            scaling = 'lin'  # default to linear scaling
            
        # Apply scaling transformation
        if scaling == 'log10':
            if data_std is not None:
                data_std = np.log10(data + data_std) - np.log10(data)
            data = np.log10(data)
            logger.info(f"Applied log10 scaling for {variable}")
        
        # Calculate or retrieve statistics
        if f'{variable}_mean' in self.dataset_stats:
            mean = self.dataset_stats[f'{variable}_mean']
            std = self.dataset_stats[f'{variable}_std']
        else:
            mean = np.mean(data)
            std = np.std(data)
            
            # Handle zero standard deviation
            if std == 0.0:
                warnings.warn(
                    f"std = 0 found for {variable}. This likely means the variable "
                    "has constant value. Dropping from dataset. Please check the data."
                )
                self.drop_X_columns.append(variable)
                std = 1.0  # prevent division by zero
                
            self.dataset_stats.update({
                f'{variable}_mean': mean,
                f'{variable}_std': std
            })
        
        # Perform normalization
        normalized_data = (data - mean) / std
        
        if data_std is not None:
            return normalized_data, data_std / std
        
        return normalized_data
    
    def unnormalize(self, data, variable, data_std = None):  
        feature_obj = None
        for prop_name, prop_obj in self.features.items():
            if variable in prop_obj.X_columns:
                feature_obj = prop_obj
                break
        
        if feature_obj is None:
            for prop_name, prop_obj in self.targets.items():
                if variable in prop_obj.X_columns:
                    feature_obj = prop_obj
                    break

        if feature_obj is None:
            raise ValueError(f"Variable {variable} not found in features or targets")
        
        mean = self.dataset_stats[f'{variable}_mean']
        std = self.dataset_stats[f'{variable}_std']
        unnormalized_data = (data * std) + mean
        print(mean, std)
        print(unnormalized_data)
    
        if feature_obj.properties["scaling"] == 'log10':
            if data_std is not None:
                data_std = np.power(10, unnormalized_data+data_std) - np.power(10, unnormalized_data)
            unnormalized_data = np.power(10, unnormalized_data)

        if data_std is not None:
            return unnormalized_data, data_std * std 
        return unnormalized_data

    def generate_candidate_space(self, **kwargs) -> pd.DataFrame:
        """Generate candidate space based on feature configurations."""
        candidates_per_feature = {}

        for feature in self.features.values():
            if isinstance(feature, CompositionFeature):
                parts = [part for part in feature.parts if part in self.X_columns]
                
                # Generate compositions
                n = kwargs.get('n_compositions', 11)
                valid_combinations = self._generate_compositions(parts, n)
                candidates_per_feature[feature.name] = pd.DataFrame(valid_combinations, columns=parts)
                
            elif isinstance(feature, DiscreteFeature):
                # Generate discrete values
                candidates = []
                for category in feature.categories:
                    encoded = [1 if cat == category else 0 for cat in feature.categories]
                    candidates.append(encoded)
                candidates_per_feature[feature.name] = candidates
                
            elif isinstance(feature, ContinuousFeature):
                # Generate continuous values
                var_range = [feature.min, feature.max]
                var_num_points = kwargs.get('n_continuous', DEFAULT_VARIABLE_DISC)
                candidates_per_feature[feature.name] = np.linspace(
                    var_range[0], var_range[1], var_num_points
                )

        # Combine all candidate features
        combined_candidates = list(itertools.product(
            *[df.values.tolist() if isinstance(df, pd.DataFrame) else df 
              for df in candidates_per_feature.values()]
        ))
        flattened_candidates = [flatten(candidate) for candidate in combined_candidates]
        return pd.DataFrame(flattened_candidates, columns=self.X_columns)

    def _generate_compositions(self, parts: List[str], n: int) -> np.ndarray:
        """Helper method to generate valid compositions."""
        def generate_combinations(dims, total=100.0, current_sum=0.0, current_comb=None):
            if current_comb is None:
                current_comb = []
            
            if dims == 1:
                last_value = round(total - current_sum, 8)
                if 0 <= last_value <= 100.0:
                    yield current_comb + [last_value]
            else:
                for i in np.linspace(0, 100.0, n)[:-1]:
                    if current_sum + i <= total:
                        yield from generate_combinations(
                            dims-1, total, current_sum + i, current_comb + [i]
                        )
        
        return np.array(list(generate_combinations(len(parts))))
    # def generate_candidate_space(self, **kwargs):
        
    #     candidates_per_feature = {}

    #     for feature in self.config['features']:
    #         feature_type = VariableType(feature['type'])
    #         if feature_type == VariableType.composition:
    #             parts = [part for part in feature['columns']['parts'] if part in self.X_columns]
                
    #             # Generate all possible combinations
    #             n = kwargs.get('n_compositions', 11)
    #             grid = np.linspace(0, 100.0, n)[:-1]
                
    #             def generate_combinations(dims, total=100.0, current_sum=0.0, current_comb=None):
    #                 if current_comb is None:
    #                     current_comb = []
                    
    #                 if dims == 1:
    #                     last_value = round(total - current_sum, 8)  # Round to avoid floating-point errors
    #                     if 0 <= last_value <= 100.0:
    #                         yield current_comb + [last_value]
    #                 else:
    #                     for i in np.linspace(0, 100.0, n)[:-1]:
    #                         if current_sum + i <= total:
    #                             yield from generate_combinations(dims-1, total, current_sum + i, current_comb + [i])
            
    #             valid_combinations = np.array(list(generate_combinations(len(parts))))
    #             valid_comb_df = pd.DataFrame(valid_combinations, columns=parts)
    #             if feature.get('design_constraints', None) is not None:
    #                 for constraint in feature['design_constraints']:
    #                     # TODO: Implement design constraints filtering
    #                     pass
    #             candidates_per_feature[feature['name']] = valid_comb_df
    #             # if kwargs.get('min_sum_constraint', None) is not None:
    #             #     constraints = kwargs.get('min_sum_constraint')
    #             #     assert isinstance(constraints, dict), "Constraint must be dict with tuple(monomers):min sum"
    #             #     for mons, min_sum in constraints.items():
    #             #         mon_ind = [self.label_x.index(m) for m in mons]
    #             #         print("mon_ind", mon_ind)
    #             #         valid_combinations = valid_combinations[np.sum(valid_combinations[:, mon_ind], axis = 1) >= min_sum]
    #         elif feature_type == VariableType.discrete:
    #             feature_obj = self.features[feature['name']]
    #             generate_type = feature['range']
    #             candidates_per_feature[feature['name']] = []
    #             for gen_type in generate_type:
    #                 candidates_per_feature[feature['name']].append(feature_obj.variable_info[feature_obj.name + "_" + gen_type]['OH_encoding'])

    #         # elif feature_type == VariableType.correlated:
    #         #     feature_obj = self.features[feature['name']]
    #         #     design_constraints = {k: v for d in feature['design_constraints'] for k, v in d.items()}
    #         #     gen_correlated_vars = design_constraints.get('design_type', feature['columns'][0]['name'])
                
    #         #     candidates_per_feature[feature['name']] = []
    #         #     for gen_type in gen_correlated_vars:
    #         #         gen_type_oh = [feature_obj.variable_info[feature_obj.name + "_type_" + gen_type]['OH_encoding']]
    #         #         # Get the correlation var 
    #         #         cor_var_dict = next((d for d in feature['columns'] if d['name'] == gen_type), None)
    #         #         if cor_var_dict is not None:
    #         #             requirements = {k: v for d in cor_var_dict['requirements'] for k, v in d.items()}
    #         #             cor_var_range = requirements['range']
    #         #             correlated_candidates = list(itertools.product(gen_type_oh,np.linspace(cor_var_range[0], cor_var_range[1], 11)))
    #         #             candidates_per_feature[feature['name']].extend([flatten(c) for c in correlated_candidates])
    #         #         else:
    #         #             raise ValueError(f"Correlated variable {gen_type} in {feature['name']} does not have requirements.")
    #         elif feature_type == VariableType.general:
    #             feature_obj = self.features[feature['name']]
    #             # get the edges of the distribution of the feature based on mean and std from the dataset stats
    #             mean = self.dataset_stats[f'{feature['name']}_mean']
    #             std = self.dataset_stats[f'{feature['name']}_std']
    #             var_range = [max(0, mean - 3*std), min(100, mean + 3*std)]
    #             var_range = feature.get('range', var_range)
    #             var_num_points = DEFAULT_VARIABLE_DISC
    #             candidates_per_feature[feature['name']] = np.linspace(var_range[0], var_range[1], var_num_points)


    #     # Combine all candidate features into a single dataframe
    #     combined_candidates = list(itertools.product(*[df.values.tolist() if isinstance(df, pd.DataFrame) else df for df in candidates_per_feature.values()]))
    #     flattened_candidates = [flatten(candidate) for candidate in combined_candidates]
    #     combined_df = pd.DataFrame(flattened_candidates, columns=self.X_columns)

    #     return combined_df
    
    def get_dataset_stats(self):
        return self.dataset_stats
    
    # TODO change the name of this function
    # def make_rec_readable(self, candidates_df):
    #     for feature_name, feature_obj in self.features.items():
    #         if isinstance(feature_obj,CorrelatedFeature) or isinstance(feature_obj, FidelityFeature):
    #             feature_columns = feature_obj.OH_columns
    #             print(feature_columns)
                
    #             candidates_df[feature_obj.name] = pd.from_dummies(candidates_df[feature_columns].astype(int), sep = "_")
    #             candidates_df.drop(columns = feature_columns, inplace = True)
    #     return candidates_df

    def make_rec_readable(self, candidates_df):
        for feature_name, feature_obj in self.features.items():
            if isinstance(feature_obj, DiscreteFeature):
                feature_columns = feature_obj.X_columns
                
                # Find the column with the maximum value (should be 1 in one-hot encoding)
                max_columns = candidates_df[feature_columns].astype(int).idxmax(axis=1)
                
                # Extract the category from the column names
                candidates_df[feature_obj.name] = max_columns.str.split('_', n=1).str[1]
                
                # Drop the one-hot encoded columns
                candidates_df.drop(columns=feature_columns, inplace=True)
        return candidates_df