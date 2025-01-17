from sci_llm.active_learning.al_loop import Active_Learning_Loop
from sci_llm.active_learning.dataset import AL_Dataset
import pandas as pd
import os
from typing import Dict


def run_active_learning(input_data : pd.DataFrame, config_dict : Dict) -> pd.DataFrame:

    for feature in config_dict['features']:
        if 'columns' not in feature.keys():
            feature['columns'] = feature['name']

    config_dict = convert_variable_types(config_dict)
    print(config_dict)
    
    dataset = AL_Dataset(config_dict)
    dataset.set_dataset(input_data)

    df = dataset.process_data()
  
    al_loop = Active_Learning_Loop(dataset)
    recs = al_loop.run_Loop()
    recs_dict = recs[monomers].to_dict('records')
    return recs_dict

def convert_variable_types(config):
    """
    Convert the variable type names to types within the active learning module
    """
    for feature in config['features']:
        if feature['type'] == 'composition':
            feature['type'] = 'composition'
        elif feature['type'] == 'continuous':
            feature['type'] = 'general'
        elif feature['type'] == 'discrete':
            feature['type'] = 'discrete'
        else:
            raise ValueError(f"Invalid feature type: {feature['type']}")
    for target in config['targets']:
        if target['target_type'] == 'regression':
            target['target_type'] = 'regression'
        elif target['target_type'] == 'classification':
            target['target_type'] = 'classification'
        else:
            raise ValueError(f"Invalid target type: {target['target_type']}")
    return config