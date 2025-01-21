import streamlit as st
import groq
import json
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from agent import *
from utils import *
from active_learning_cycle import *

load_dotenv('.env')

# Initialize Groq client
client = client = groq.Client(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
#Initialize json dict
if 'json_dict' not in st.session_state:
    st.session_state.json_dict = None
    
# Inittialize the input data
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# Streamlit app
st.title("Scientific Assistant Chatbot")

# Function to generate response
def generate_response(messages: List[Dict[str, str]]) -> str:
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=1000,
        top_p=1,
        stream=False,
        stop=None
    )
    return chat_completion.choices[0].message.content


# Load the base prompt
with open('prompts/base_prompt.txt', 'r') as file:
  base_prompt = file.read()

with open('prompts/parsing_vars.txt', 'r') as file:
  parsing_vars = file.read()



# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("What's on your mind?"):

    # Add base prompt to the session state if it's the first input
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "system", "content": base_prompt})

    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "system", "content": parsing_vars})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Check if response contains JSON
            if "```json" in response:
                json_dict = extract_json_from_llm_output(response)
                if json_dict:
                    st.session_state.json_dict = set_json_display(json_dict, response)
                else:
                    st.error("Invalid JSON format detected. Please retry.")
                    st.markdown(response)
            else:
                st.markdown(response)

    # Display the user message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)                

# Sidebar header
with st.sidebar:
    st.header("Workflow")
    
# Sidebar with chat history clear button
with st.sidebar:
    if st.button("Restart"):
        st.session_state.messages = []
        st.stop()
    
# Test Saying step 1, provide a description of research problem
with st.sidebar:
    st.markdown("### Step 1: Describe Your Research Problem")
    st.markdown("Provide a brief description of your research problem.")


# Only show the rest when the subsequent steps are triggered
if st.session_state.json_dict is not None:
    # Sidebar button for prepping data
    with st.sidebar:
        st.markdown("### Step 2: Prepare Dataset")
        if st.button("Prepare Dataset"):
            if st.session_state.json_dict is None:
                st.warning("Please provide information about your research problem first.")
            else:
                formatted_data = prep_empty_df_from_json(st.session_state.json_dict)
                                    # Display the empty DataFrame
                st.markdown("### Empty DataFrame Based on Structured Response:")
                st.dataframe(formatted_data)

                # Provide download link for the DataFrame as a CSV
                csv = formatted_data.to_csv(index=False)
                b64 = st.download_button(
                    label="Download Empty DataFrame as CSV",
                    data=csv,
                    file_name="data_format.csv",
                    mime="text/csv"
                )

    # Sidebar button to trigger processing
    with st.sidebar:
        st.markdown("### Step 3: Upload Formatted Data")
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read the uploaded file into a pandas DataFrame
            st.session_state.input_data = pd.read_csv(uploaded_file)
            try:
                check_uploaded_data_columns(st.session_state.input_data, st.session_state.json_dict)
            except DatasetException as e:
                st.warning(f"Error in uploaded data: {e}")
                st.session_state.input_data = None
            
            st.write("Data uploaded successfully!")
        else:
            st.session_state.input_data = None

if st.session_state.input_data is not None:
    with st.sidebar:
        if st.button("Suggest Experiments"):
            recommendations = run_active_learning(
                st.session_state.input_data,
                st.session_state.json_dict
            )
            # Store the DataFrame in session_state
            st.session_state.recommendations = recommendations

# Display the DataFrame if we have it
if st.session_state["recommendations"] is not None:
    st.write("### Experiment Recommendations")
    st.dataframe(st.session_state.recommendations)