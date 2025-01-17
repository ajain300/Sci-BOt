import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv('.env')

def initialize_llm():
    return ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

def set_structured_llm(llm : ChatGroq):
    struct_llm = llm.with_structured_output(create_json_schemas())
    prompt = get_prompt()
    struct_llm_template = prompt | struct_llm
    return struct_llm

class Active_Learning_Config(BaseModel):
    Variable: str = Field(description="The variable for the active learning")
    Parts: list = Field(description="The different choices we have for the variable.")

class Clarify(BaseModel):
    Clarify: str = Field(description="The clarification for the user.")

def create_json_schemas():
    # Define your JSON schemas here
    json_schema_simple = {
        "title": "active_learning_config",
        "description": "Template for each active learning variable.",
        "type": "object",
        "properties": {
            "Variables": {
                "type": "string",
                "description": "Variable for active learning",
            },
            "Parts": {
                "type": "array",
                "description": "The different choice labels we have for the variable.",
            },
            "Ranges": {
                "type": "array",
                "description": "The possible ranges for each part.",
            }
        },
        "required": ["Variable", "Parts"],
    }
    # Add other schemas as needed
    return json_schema_simple

@tool
def run_active_learning(config):
    """
    Runs active learning for a given configuration.
    """
    return [[0, 0, 0, 0]]

def get_prompt():
    return ChatPromptTemplate.from_messages([("system", """
        You are a helpful scientific assistant. Your job is to give generate a configuration for an active learning experiment. 
        Return a configuration for an active learning experiment with the information on the features (independent variables) and targets (dependent variables).
        The features can be of type composition, continuous or discrete. The targets can be of type regression or classification.
        For composition, the ratios should default between 0 and 1.0.
        Respond only in json format.
        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")]
    )