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
from .features import *
from .targets import *
import warnings

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
        variable_info (dict): Dictionary to store variable information.
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
    minimum_feature_keys = ['name', 'type', 'columns']
    minimum_target_keys = ['name', 'type', 'scaling', 'unit']

    def __init__(self, data_config : Union[str, Dict], **kwargs):
        
        # setup logger
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        
        # Setup config and requirements
        if isinstance(data_config, dict):
            self.config = data_config
        else:
            with open(data_config, 'r') as file:
                self.config = yaml.safe_load(file)
        
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

    def read_dataset(self, path : str):
        self.wide_dataset = pd.read_excel(path, keep_default_na=True)

    def set_dataset(self, dataset : pd.DataFrame):
        self.wide_dataset = dataset
    
    @property
    def X_columns(self):
        return [col for feat in self.features.values() for col in feat.X_columns if not isinstance(feat, TargetFeature)]

    @property
    def X_columns_pred(self):
        return [col for feat in self.features.values() for col in feat.X_columns if not isinstance(feat, TargetFeature) and col not in self.drop_X_columns]

    def process_data(self) -> pd.DataFrame:
        """
        Process data based on the configuration provided, handling different types of
        features and their requirements.
        
        :return: Processed DataFrame
        :rtype: pd.DataFrame
        """
        data = self.wide_dataset

        # PROCESS FEATURES
        for i, feature in enumerate(self.config['features']):
            # Check if bare minimum keys are present in the feature config
            if feature.get('type', None) == VariableType.target.value:
                missing_keys = [key for key in self.minimum_feature_keys if key not in feature and key != 'design_constraints']
            else:
                missing_keys = [key for key in self.minimum_feature_keys if key not in feature]

            if missing_keys:
                if 'name' in missing_keys:
                    raise ValueError(f"Missing 'name' key in {i}th target feature")
                else:
                    raise ValueError(f"Missing required values for feature {feature['name']}: {missing_keys}")

            feature_type = VariableType(feature['type'])

            # Extract column information
            if feature_type == VariableType.composition:
                # Columns are the actual parts of the composition
                columns = feature['columns']['parts']

                # Ranges are a dict defined for each monomer. If not given, set to default: [0, 100]
                ranges = feature['columns']['range']
                for part in columns:
                    if part not in ranges:
                        ranges[part] = [0, 100]

                self.X_columns.extend(columns)
                variable_info = {'scaling': feature.get('scaling', 'lin'), 'ranges': ranges}
                comp = CompositionFeature(feature, 
                                          X_columns=columns, 
                                          variable_info=variable_info)
                comp.process()
                self.features[feature['name']] = comp
            elif feature_type == VariableType.correlated:
                columns = feature['columns']
                # Don't add the correlated columns to the X_columns, this will be don in the processing step
                corr = CorrelatedFeature(feature,
                                         initial_columns=columns)
                corr.process()
                self.features[feature['name']] = corr
            elif feature_type == VariableType.discrete:
                columns = feature['columns']
                # Check if fidelities are given
                if 'range' not in feature:
                    raise ValueError(f"Need to define the fidelities for {feature['name']} by adding 'fidelities' key to feature list")
                # Add the allowed values constraint explicitly, which is implicitly defined as the fidelities
                # feature['requirements'].append({'allowed_values': feature['fidelities']})
                disc = DiscreteFeature(feature, 
                                           initial_columns=columns)
                disc.process()
                self.features[feature['name']] = disc
            else:
                general = GeneralFeature(feature, 
                                         initial_columns=feature['columns'])
                general.process()
                self.features[feature['name']] = general
                columns = feature['columns']
                self.X_columns.extend(columns)


            # Process feature-level requirements
            self._process_requirements(data, feature_type, columns, feature.get('requirements', []))

            # Process column-specific requirements
            if isinstance(columns, dict):
                for col_type, col_info in columns.items():
                    if isinstance(col_info, dict) and 'requirements' in col_info:
                        self._process_requirements(data, feature_type, col_info['name'], col_info['requirements'])
            elif isinstance(columns, list):
                for col in columns:
                    if isinstance(col, dict) and 'requirements' in col:
                        self._process_requirements(data, feature_type, col['name'], col['requirements'])

            # Process the feature based on its type
            if feature_type != VariableType.target:
                self.logger.info(f"Processing feature: {feature['name']}, type {feature_type}")
                print("columns going into _process_feature_data", columns)
                data = self._process_feature_data(data, feature_type, columns, feature['name'])

        # PROCESS TARGETS
        for i, target in enumerate(self.config['targets']):
            # Check if all required keys are present in the feature dict
            required_keys = self.minimum_target_keys
            missing_keys = [key for key in required_keys if key not in target]
            
            if missing_keys:
                raise ValueError(f"Missing required values for target {target['name']}: {missing_keys}")

            # Set property info based on config
            columns = {
                'value': target['name'],
                'std': f"{target['name']}_std"
            }
            prop_name = target['name']
            properties = {
                'type': target['type']['value'],
                'goal': target['type']['goal'],
                'value_col': list(columns.values())[0],
                'std_col': list(columns.values())[1],
                'scaling': target['scaling'],
                'unit': target['unit'],
                'requirements': target.get('requirements', [])
            }
            variable_info = {'scaling': target.get('scaling', 'lin')}
            target_obj = TargetFeature(target, 
                                    X_columns=list(columns.values()),
                                    target_type = target['type']['value'],
                                    target_goal = target['type']['goal'],
                                    variable_info=variable_info, 
                                    properties=properties)
            target_obj.process()
            self.targets[prop_name] = target_obj
        
        data = self._process_targets(data)
        data = self._normalize_properties(data)
        # Get property stats after processing data and checking for issues
        self.input_data = data
        return data

    def _process_feature_data(self, data: pd.DataFrame, feature_type: VariableType, columns: Any, name : str) -> pd.DataFrame:
        if feature_type == VariableType.correlated:
            feature : CorrelatedFeature = self.features[name]
            columns = [col['name'] for col in columns]
            corr_type_col = f"{name}_type"
            corr_amt_col = f"{name}_amount"
            feature.X_columns = [corr_type_col, corr_amt_col]
            dataset_cols = [corr_type_col, corr_amt_col]
            
            # Add scaling info to columns
            # for feature in self.config['features']:
            #     if feature['name'] == name: 
            #         self.variable_info[corr_type_col] = {'scaling': feature.get('scaling', 'lin')}
            #         self.variable_info[corr_amt_col] = {'scaling' : feature.get('scaling', 'lin')}

            # Function to organize the correlated samples
            def create_correlated_samples(row, columns):
                base_row = row.copy()
                samples = []
                correlated_num = len(columns)
                for i, corr_col in enumerate(columns):
                    if corr_col in row and not pd.isna(row[corr_col]):
                        cure_row = base_row.copy()
                        cure_row[corr_type_col] = corr_col
                        cure_row[corr_amt_col] = row[corr_col]
                        samples.append(cure_row)
                return samples if samples else [base_row]
                
            new_rows = []
            for _, row in data.iterrows():
                new_rows.extend(create_correlated_samples(row, columns))
            data = pd.DataFrame(new_rows)
            data = data.drop(columns=columns)

            # Apply one-hot encoding to the correlated type column
            data = pd.get_dummies(data, columns=[corr_type_col], prefix=corr_type_col)
            
            # Ensure the one-hot encoded columns are of integer type
            one_hot_columns = [col for col in data.columns if col.startswith(corr_type_col)]
            data[one_hot_columns] = data[one_hot_columns].astype(int)

            # Update feature.X_columns and dataset_cols to include the new one-hot encoded columns
            feature.X_columns = one_hot_columns + [corr_amt_col]
            feature.OH_columns = one_hot_columns
            dataset_cols = one_hot_columns + [corr_amt_col]

            for i, col in enumerate(one_hot_columns):
                feature.variable_info[col] = {'scaling': 'lin', 'OH_encoding': [1 if i == idx else 0 for idx in range(len(one_hot_columns))]}
        

            feature.variable_info[corr_amt_col] = {'scaling': feature.feature_config.get('scaling', 'lin')}

        elif feature_type == VariableType.discrete:
            # Get the discrete types, convert to one-hot encoding
            dataset_cols = []
            feature : DiscreteFeature = self.features[name]
            discrete_types = feature.feature_config['range']
            col = feature.initial_columns
            
            # Apply one-hot encoding to the correlated type column
            data = pd.get_dummies(data, columns=[col], prefix=name)
            
            # Ensure the one-hot encoded columns are of integer type
            one_hot_columns = [col for col in data.columns if col.startswith(name)]
            data[one_hot_columns] = data[one_hot_columns].astype(int)

            # Update feature.X_columns and dataset_cols to include the new one-hot encoded columns
            feature.X_columns = one_hot_columns
            feature.OH_columns = one_hot_columns
            dataset_cols = one_hot_columns

            for i, col in enumerate(one_hot_columns):
                feature.variable_info[col] = {'scaling': 'lin', 'OH_encoding': [1 if i == idx else 0 for idx in range(len(one_hot_columns))]}
        elif feature_type == VariableType.general:
            if isinstance(columns, dict):
                dataset_cols = [columns['name'],]
            elif isinstance(columns, list):
                dataset_cols = columns
            else:
                dataset_cols = [columns]
            
            feature: GeneralFeature = self.features[name]

            feature.X_columns = dataset_cols
            feature.variable_info[dataset_cols[0]] = {'scaling': feature.feature_config.get('scaling', 'lin')}
        else:
            dataset_cols = columns
        # print(columns)
        # print(dataset_cols)
        for col in dataset_cols:
            if data[col].nunique() == 1:
                data = data.drop(columns=col)
                # self.X_columns = [x_col for x_col in self.X_columns if col != x_col]
                # self.variable_info[col]['dropped'] = True
            else:
                self.feature_columns[name] = dataset_cols if isinstance(dataset_cols, list) else dataset_cols.keys()
        return data

    def _process_targets(self, data: pd.DataFrame):
        # Melting the DataFrame
        print(data)
        
        value_columns = [target.properties['value_col'] for target in self.targets.values()]
        std_columns = [target.properties['std_col'] for target in self.targets.values()]
   
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
        print(data)
        
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
        
    # Helper
    def normalize(self, data : np.ndarray, variable : str, data_std = None):
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

        if feature_obj is None:
            raise ValueError(f"Variable {variable} not found in features or targets")
        
        if feature_obj.variable_info[variable].get('scaling', 'lin') == 'log10':
            if data_std is not None:
                data_std = np.log10(data+data_std) - np.log10(data)
            data = np.log10(data)
            self.logger.info(f"log10 scaling for {variable}")
        
        if f'{variable}_mean' in self.dataset_stats.keys():
            mean = self.dataset_stats[f'{variable}_mean']
            std = self.dataset_stats[f'{variable}_std']
        else:
            mean = np.mean(data)
            std = np.std(data)
            # Check for 0.0 std values in the dataset
            if std == 0.0:
                warnings.warn(f"std = 0 found upon evaluation of {variable}. This is likely due to missing values in the dataset and means that variable is the same number. Dropping {variable} from dataset. Please check the dataset.")
                self.drop_X_columns.append(variable)
                
            self.dataset_stats.update({f'{variable}_mean': mean, f'{variable}_std': std}) # type: ignore
        normalized_data = (data - mean) / std

        # Save the mean and std for later use
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

    def generate_candidate_space(self, **kwargs):
        
        candidates_per_feature = {}

        for feature in self.config['features']:
            feature_type = VariableType(feature['type'])
            if feature_type == VariableType.composition:
                parts = [part for part in feature['columns']['parts'] if part in self.X_columns]
                
                # Generate all possible combinations
                n = kwargs.get('n_compositions', 11)
                grid = np.linspace(0, 100.0, n)[:-1]
                
                def generate_combinations(dims, total=100.0, current_sum=0.0, current_comb=None):
                    if current_comb is None:
                        current_comb = []
                    
                    if dims == 1:
                        last_value = round(total - current_sum, 8)  # Round to avoid floating-point errors
                        if 0 <= last_value <= 100.0:
                            yield current_comb + [last_value]
                    else:
                        for i in np.linspace(0, 100.0, n)[:-1]:
                            if current_sum + i <= total:
                                yield from generate_combinations(dims-1, total, current_sum + i, current_comb + [i])
            
                valid_combinations = np.array(list(generate_combinations(len(parts))))
                valid_comb_df = pd.DataFrame(valid_combinations, columns=parts)
                if feature.get('design_constraints', None) is not None:
                    for constraint in feature['design_constraints']:
                        # TODO: Implement design constraints filtering
                        pass
                candidates_per_feature[feature['name']] = valid_comb_df
                # if kwargs.get('min_sum_constraint', None) is not None:
                #     constraints = kwargs.get('min_sum_constraint')
                #     assert isinstance(constraints, dict), "Constraint must be dict with tuple(monomers):min sum"
                #     for mons, min_sum in constraints.items():
                #         mon_ind = [self.label_x.index(m) for m in mons]
                #         print("mon_ind", mon_ind)
                #         valid_combinations = valid_combinations[np.sum(valid_combinations[:, mon_ind], axis = 1) >= min_sum]
            elif feature_type == VariableType.discrete:
                feature_obj = self.features[feature['name']]
                generate_type = feature['range']
                candidates_per_feature[feature['name']] = []
                for gen_type in generate_type:
                    candidates_per_feature[feature['name']].append(feature_obj.variable_info[feature_obj.name + "_" + gen_type]['OH_encoding'])

            # elif feature_type == VariableType.correlated:
            #     feature_obj = self.features[feature['name']]
            #     design_constraints = {k: v for d in feature['design_constraints'] for k, v in d.items()}
            #     gen_correlated_vars = design_constraints.get('design_type', feature['columns'][0]['name'])
                
            #     candidates_per_feature[feature['name']] = []
            #     for gen_type in gen_correlated_vars:
            #         gen_type_oh = [feature_obj.variable_info[feature_obj.name + "_type_" + gen_type]['OH_encoding']]
            #         # Get the correlation var 
            #         cor_var_dict = next((d for d in feature['columns'] if d['name'] == gen_type), None)
            #         if cor_var_dict is not None:
            #             requirements = {k: v for d in cor_var_dict['requirements'] for k, v in d.items()}
            #             cor_var_range = requirements['range']
            #             correlated_candidates = list(itertools.product(gen_type_oh,np.linspace(cor_var_range[0], cor_var_range[1], 11)))
            #             candidates_per_feature[feature['name']].extend([flatten(c) for c in correlated_candidates])
            #         else:
            #             raise ValueError(f"Correlated variable {gen_type} in {feature['name']} does not have requirements.")
            elif feature_type == VariableType.general:
                feature_obj = self.features[feature['name']]
                # get the edges of the distribution of the feature based on mean and std from the dataset stats
                mean = self.dataset_stats[f'{feature['name']}_mean']
                std = self.dataset_stats[f'{feature['name']}_std']
                var_range = [max(0, mean - 3*std), min(100, mean + 3*std)]
                var_range = feature.get('range', var_range)
                var_num_points = DEFAULT_VARIABLE_DISC
                candidates_per_feature[feature['name']] = np.linspace(var_range[0], var_range[1], var_num_points)


        # Combine all candidate features into a single dataframe
        combined_candidates = list(itertools.product(*[df.values.tolist() if isinstance(df, pd.DataFrame) else df for df in candidates_per_feature.values()]))
        flattened_candidates = [flatten(candidate) for candidate in combined_candidates]
        combined_df = pd.DataFrame(flattened_candidates, columns=self.X_columns)

        return combined_df
    
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
            if isinstance(feature_obj, CorrelatedFeature) or isinstance(feature_obj, DiscreteFeature):
                feature_columns = feature_obj.OH_columns
                
                # Find the column with the maximum value (should be 1 in one-hot encoding)
                max_columns = candidates_df[feature_columns].astype(int).idxmax(axis=1)
                
                # Extract the category from the column names
                candidates_df[feature_obj.name] = max_columns.str.split('_', n=1).str[1]
                
                # Drop the one-hot encoded columns
                candidates_df.drop(columns=feature_columns, inplace=True)
        return candidates_df