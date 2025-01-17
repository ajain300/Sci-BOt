import json
import streamlit as st
import pandas as pd
from sci_llm.utils.exceptions import *

def extract_json_from_llm_output(text):
    """
    Extracts JSON from text.
    """
    # Find the JSON object in the text
    json_start = text.find("```json")
    json_end = text.rfind("```")
    json_text = text[json_start + 7:json_end]
    print(json_text)
    
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        st.error("Invalid JSON format.")
        return None

    return data

def plain_llm_output_remove_json(text):
    """
    Extracts JSON from text.
    """
    # Find the JSON object in the text
    json_start = text.find("```json")
    text = text[:json_start]

    return text

def prep_empty_df_from_json(json_dict):
    """
    Prepares an empty DataFrame from the JSON dictionary.
    """
    # Get the columns from the JSON
    # Extract feature names
    feature_names = []
    for feature in json_dict['features']:
        if feature['type'] == 'composition':
            feature_names.extend(feature['columns'])
        else:
            feature_names.append(feature['name'])

    # Extract target names
    target_names = [target['name'] for target in json_dict['targets']]
    
    # Create an empty DataFrame
    df = pd.DataFrame(columns= feature_names + target_names)
    return df

def check_uploaded_data_columns(uploaded_data, json_dict):
    """
    Checks if the uploaded data columns match the JSON dictionary.
    """
    # Get the columns from the JSON
    # Extract feature names
    feature_names = []
    for feature in json_dict['features']:
        if feature['type'] == 'composition':
            feature_names.extend(feature['columns'])
        else:
            feature_names.append(feature['name'])

    # Extract target names
    target_names = [target['name'] for target in json_dict['targets']]
    
    # Check if the columns match
    if set(uploaded_data.columns) != set(feature_names + target_names):
        raise ColumnMismatchError(f"Columns in the uploaded data do not match the JSON dictionary. Expected {feature_names + target_names}, got {uploaded_data.columns}")