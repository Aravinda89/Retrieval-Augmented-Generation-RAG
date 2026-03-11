import requests
import json
from dateutil import parser
import pandas as pd
from pprint import pprint as original_pprint
import os
from together import Together
import numpy as np
from typing import List, Dict

def get_proxy_url():
    """
    Get the proxy URL from environment variable or fall back to Together.ai endpoint.
    Uses TOGETHER_BASE_URL environment variable set in Dockerfile.
    Defaults to https://api.together.xyz/ if not set.
    """
    return os.environ.get('TOGETHER_BASE_URL', 'https://api.together.xyz/')

def get_proxy_headers():
    """
    Get the appropriate headers for API calls based on the platform.
    Returns Authorization header with Together API key if available.
    """
    return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}

def get_together_key():
    """
    Get the Together API key from environment variables.
    """
    return os.environ.get("TOGETHER_API_KEY", "")


# Distance formulas. 
# In this ungraded lab, distance formulas will be implemented here. However, in future assignments, you will import functions from specialized libraries.
def cosine_similarity(v1, array_of_vectors):
    """
    Compute the cosine similarity between a vector and an array of vectors.
    
    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of cosine similarities between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store similarities
    similarities = []
    
    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]
    
    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Compute the dot product of v1 and v2
        dot_product = np.dot(v1, v2)
        # Compute the norms of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # Compute the cosine similarity and append to the list
        similarity = dot_product / (norm_v1 * norm_v2)
        similarities.append(similarity)
    return np.array(similarities)

def euclidean_distance(v1, array_of_vectors):
    """
    Compute the Euclidean distance between a vector and an array of vectors.
    
    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of Euclidean distances between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store distances
    distances = []
    
    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]
    
    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Check if the input arrays have the same shape
        if v1.shape != v2.shape:
            raise ValueError(f"Shapes don't match: v1 shape: {v1.shape}, v2 shape: {v2.shape}")
        # Calculate the Euclidean distance and append to the list
        dist = np.sqrt(np.sum((v1 - v2) ** 2))
        distances.append(dist)
    return distances


def format_date(date_string):
    # Parse the input string into a datetime object
    date_object = parser.parse(date_string)
    # Format the date to "YYYY-MM-DD"
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date

def pprint(*args, **kwargs):
    kwargs.setdefault('sort_dicts', False)
    original_pprint(*args, **kwargs)
    
def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                               together_api_key = None,
                              **kwargs):

    # Remove None parameters for Together API - don't set to string 'none'
    if top_p is None:
        payload_top_p = None
    else:
        payload_top_p = top_p
    if temperature is None:
        payload_temperature = None
    else:
        payload_temperature = temperature

    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
        **kwargs
    }
    # Only add temperature and top_p if they're not None
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        url = os.path.join(get_proxy_url(), 'v1/chat/completions')
        response = requests.post(url, json = payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key =  together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict


def generate_with_multiple_input(messages: List[Dict],
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                                together_api_key = None,
                                **kwargs):
    # Remove None parameters for Together API
    if top_p is None:
        payload_top_p = None
    else:
        payload_top_p = top_p
    if temperature is None:
        payload_temperature = None
    else:
        payload_temperature = temperature

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs
    }
    # Only add temperature and top_p if they're not None
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        url = os.path.join(get_proxy_url(), 'v1/chat/completions')
        response = requests.post(url, json = payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key =  together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict

# Read the CSV without parsing dates

def read_dataframe(path):
    df = pd.read_csv(path)

    # Apply the custom date formatting function to the relevant columns
    df['published_at'] = df['published_at'].apply(format_date)
    df['updated_at'] = df['updated_at'].apply(format_date)

    # Convert the DataFrame to dictionary after formatting
    df= df.to_dict(orient='records')
    return df



import ipywidgets as widgets
from IPython.display import display, Markdown

def display_widget(llm_call_func, semantic_search_retrieve, bm25_retrieve, reciprocal_rank_fusion):
    def on_button_click(b):
        query = query_input.value
        top_k = slider.value
        # Clear existing outputs
        for output in [output1, output2, output3, output4]:
            output.clear_output()
        status_output.clear_output()
        # Display "Generating..." message
        status_output.append_stdout("Generating...\n")
        # Update outputs one by one
        results = [
            (output1, llm_call_func, query, True, top_k, semantic_search_retrieve),
            (output2, llm_call_func, query, True, top_k, bm25_retrieve),
            (output3, llm_call_func, query, True, top_k, reciprocal_rank_fusion),
            (output4, llm_call_func, query, False, top_k, None)
        ]
        for output, func, query, use_rag, top_k, retriever in results:
            response = func(query=query, use_rag=use_rag, top_k=top_k, retrieve_function=retriever)
            with output:
                display(Markdown(response))
        # Clear "Generating..." message
        status_output.clear_output()
        
    query_input = widgets.Text(
        description='',
        placeholder='Type your query here',
        layout=widgets.Layout(width='100%')
    )
    
    slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Top K:',
        style={'description_width': 'initial'}
    )
    
    output_style = {'border': '1px solid #ccc', 'width': '100%'}
    output1 = widgets.Output(layout=output_style)
    output2 = widgets.Output(layout=output_style)
    output3 = widgets.Output(layout=output_style)
    output4 = widgets.Output(layout=output_style)
    status_output = widgets.Output()

    submit_button = widgets.Button(
        description="Get Responses",
        style={'button_color': '#eee', 'font_color': 'black'}
    )
    submit_button.on_click(on_button_click)

    label1 = widgets.Label(value="Semantic Search")
    label2 = widgets.Label(value="BM25 Search")
    label3 = widgets.Label(value="Reciprocal Rank Fusion")
    label4 = widgets.Label(value="Without RAG")

    display(widgets.HTML("""
    <style>
        .custom-output {
            background-color: #f9f9f9;
            color: black;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .widget-text, .widget-button {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ddd !important;
        }
        .widget-output {
            background-color: #f9f9f9 !important;
            color: black !important;
        }
        input[type="text"] {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ddd !important;
        }
    </style>
    """))
    
    display(query_input, slider, submit_button, status_output)
    
    # Create individual vertical containers for each label and output
    vbox1 = widgets.VBox([label1, output1], layout={'width': '45%'})
    vbox2 = widgets.VBox([label2, output2], layout={'width': '45%'})
    vbox3 = widgets.VBox([label3, output3], layout={'width': '45%'})
    vbox4 = widgets.VBox([label4, output4], layout={'width': '45%'})
    
    # HBoxes to arrange two VBoxes in each row
    hbox_outputs1 = widgets.HBox([vbox1, vbox2], layout={'justify_content': 'space-between'})
    hbox_outputs2 = widgets.HBox([vbox3, vbox4], layout={'justify_content': 'space-between'})

    def style_outputs(*outputs):
        for output in outputs:
            output.layout.margin = '5px'
            output.layout.height = '300px'
            output.layout.padding = '10px'
            output.layout.overflow = 'auto'
            output.add_class("custom-output")
            
    style_outputs(output1, output2, output3, output4)
    
    # Display two rows with two outputs each
    display(hbox_outputs1)
    display(hbox_outputs2)

