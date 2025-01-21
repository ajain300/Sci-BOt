from sci_llm.active_learning_cycle import run_active_learning
import pandas as pd
import json

# Sample function to be tested
def test_run_active_learning():
    input_data = pd.read_csv('dataset/data_format_example.csv')
    with open('dataset/json_example.json', 'r') as f:
        config_dict = json.load(f)
    # {'features': [{'name': 'Monomer_Composition', 'columns': ['PEG', 'IBOA', 'IDA'], 'type': 'composition', 'range': {'PEG': [0, 100], 'IBOA': [0, 50], 'IDA': [0, 100]}}, {'name': 'Curing_percentage', 'range': [0, 100], 'type': 'continuous', 'columns': 'Curing_percentage'}, {'name': 'Experiment_type', 'range': ['Experiment', 'Simulation'], 'type': 'discrete', 'columns': 'Experiment_type'}], 'targets': [{'name': "Young's_modulus", 'target_type': 'regression', 'scaling': 'lin', 'goal': 'MAX', 'unit': 'Pa'}, {'name': 'Tg', 'target_type': 'regression', 'scaling': 'lin', 'goal': 'MAX', 'unit': '°C'}], 'recommendation_method': 'Bayesian_Optimization'}
    recs = run_active_learning(input_data, config_dict)
    print(recs)