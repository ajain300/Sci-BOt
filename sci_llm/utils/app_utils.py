import json
import streamlit as st
from typing import Dict
from .data_utils import *


# Handle json output from agent
def set_json_display(json_dict: Dict, response: str):
    # Display the plain response in the app
    response = plain_llm_output_remove_json(response)
    st.markdown(response)
    
    # Initialize session state for storing the JSON if not already set
    if 'adjusted_response' not in st.session_state:
        st.session_state.adjusted_response = json.dumps(json_dict, indent=2)
        st.session_state.json_submitted = False  # To track if the button was pressed

    # Display the JSON editor for user to make adjustments
    st.markdown("### Generated Configuration:")
    json_editor = st.text_area(
        "I've come up with a configuration that we can use to optimize your experiments:",
        st.session_state.adjusted_response, 
        height=300
    )
    
    # Button to submit the adjusted JSON
    if st.button("Submit Configuration", key="submit_json"):
        try:
            # Parse the edited JSON
            adjusted_response = json.loads(json_editor)
            # Update session state with the new value
            st.session_state.adjusted_response = json.dumps(adjusted_response, indent=2)
            st.session_state.json_submitted = True  # Mark that the JSON was submitted
            st.success("Adjusted configuration submitted.")
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please correct and try again.")
    
    # Handle logic after the JSON is submitted
    if st.session_state.json_submitted:
        return st.session_state.adjusted_response
    else:
        return json_dict

# Handle json output from agent
# def set_json_display(json_dict : Dict, response):
#     # Check if the response is a JSON
#     # try:
#     #     response_data = json.loads(response)
#     #     is_json = True
#     # except json.JSONDecodeError:
#     #     is_json = False

#     # Use Streamlit columns to create a layout with JSON on the right
#     # left_column, right_column = st.columns([1, 2])  # 1:2 ratio makes the right column wider
    
#     response = plain_llm_output_remove_json(response)
    
#     # with left_column:
#         # Continue displaying conversation in the left column
#     st.markdown(response)

#     # with right_column:
#     st.markdown("### Generated Configuration:")
#     json_editor = st.text_area("I've come up with a configuration that we can use to optimize your experiments:", json.dumps(json_dict, indent=2), height=300)
    
#     # Submit button to confirm adjustments
#     if st.button("Submit Configuration", key="submit_json"):
#         try:
#             # Parse adjusted JSON
#             adjusted_response = json.loads(json_editor)
#             st.session_state.adjusted_response = json.dumps(adjusted_response, indent=2)
#             print("submitting adjusted response")
#             st.success("Adjusted JSON submitted.")
#         except json.JSONDecodeError:
#             st.error("Invalid JSON format. Please correct and try again.")
                
#         return st.session_state.adjusted_response
#     else:
#         print("printing json dict")
#         return json_dict