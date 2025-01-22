# Sci-Bot: Augmenting Bayesian Optimization and Active Learning with LLM Frontend

Sci-Bot combines Bayesian optimization and active learning with a large language model (LLM) frontend to create a seamless and efficient workflow for scientific exploration.

---

## Installation

### Prerequisites
Make sure you have [Poetry](https://python-poetry.org/) installed. You can install it using pip:

```bash
pip install poetry
```

### Setting Up the Environment
1. In the root directory of the repository, set the Python version to use:
   ```bash
   poetry env use /path/to/python3.12
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```

Dependencies are specified in `pyproject.toml` and will be automatically resolved by Poetry.

---

## Repository Structure

The codebase is divided into two main parts:

### 1. **`sci_llm`**
This folder contains the Streamlit-based frontend application. It integrates an LLM to:
- Automatically generate the JSON configuration based on user prompts.
- Accept user input data formatted as specified in the generated JSON.
- Pass the data and configuration to `active_learning` for inference.

---

### 2. **`sci_llm/active_learning`**
This folder contains the backend for training machine learning models and making recommendations via inference.

- **Requirements**:
  - A **tabular dataset**.
  - A **JSON configuration** describing the data space.

> **Note**: The column names in the dataset **must match exactly** with those specified in the JSON configuration.
An example of a json configuration and corresponding dataset are given in the `dataset` folder.

---

## Helper Functions

The `active_learning` backend can be accessed through a set of helper functions provided in `sci_llm/active_learning_cycle.py`:

### Main Functions:
1. **`run_active_learning()`**  
   - The primary function to interface with `active_learning`.

2. **`modify_configuration()`**  
   - Adapts the frontend-generated configuration to the format required by `active_learning`.

3. **`modify_input_data()`**  
   - Converts user input data into the format expected by `active_learning`.

> **Note**: The `modify_*` functions may need adjustments if a new frontend or LLM is integrated.

---

## Using the Application

### Steps to Run the App:
1. Navigate to the `sci_llm` folder:
   ```bash
   cd sci_llm
   ```
2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
