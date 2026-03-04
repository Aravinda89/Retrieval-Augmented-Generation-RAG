import json
import numpy as np
import pandas as pd
from pprint import pprint as original_pprint
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os 
from together import Together

model_name = os.path.join(os.environ['MODEL_PATH'], "BAAI/bge-base-en-v1.5")

model = SentenceTransformer(model_name)

EMBEDDINGS = joblib.load("embeddings.joblib")

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

def pprint(*args, **kwargs):
    print(json.dumps(*args, indent = 2))

def format_date(date_string):
    # Parse the input string into a datetime object
    date_object = parser.parse(date_string)
    # Format the date to "YYYY-MM-DD"
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date

# Read the CSV without parsing dates

def read_dataframe(path):
    df = pd.read_csv(path)

    # Apply the custom date formatting function to the relevant columns
    df['published_at'] = df['published_at'].apply(format_date)
    df['updated_at'] = df['updated_at'].apply(format_date)

    # Convert the DataFrame to dictionary after formatting
    df= df.to_dict(orient='records')
    return df


def generate_with_single_input(prompt: str, 
                               role: str = 'assistant', 
                               top_p: float = None, 
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                               together_api_key = None,
                              **kwargs):
    
    if top_p is None:
        top_p = 'none'
    if temperature is None:
        temperature = 'none'

    payload = {
            "model": model,
            "messages": [{'role': role, 'content': prompt}],
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
                  }
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        url = os.path.join(os.environ.get("TOGETHER_BASE_URL", "https://proxy.dlai.link/coursera_proxy/together"), 'v1/chat/completions')   
        response = requests.post(url, json = payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: f{response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        if top_p == 'none':
            payload['top_p'] = None
        if temperature == 'none':
            payload['temperature'] = None
        client = Together(api_key =  together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict



def concatenate_fields(dataset, fields):
    # Initialize the list where the texts will be stored    
    concatenated_data = [] 

    # Iterate over movies
    for data in dataset:
        # Initialize text as an empty string
        text = "" 

        # Iterate over the fields
        for field in fields: 
            # Get the desired field (if the key is missing an empty string should be used)
            context = data.get(field, '') 

            if context:
                # Add the context to the text (add an extra space so fields are separate)
                text += f"{context} " 

        # Strip whitespaces from the text
        text = text.strip()[:493]
        # Append the text with extra context to the list
        concatenated_data.append(text) 
    
    return concatenated_data




NEWS_DATA = pd.read_csv("./news_data_dedup.csv").to_dict(orient = 'records')


def retrieve(query, top_k = 5):
    query_embedding = model.encode(query)

    similarity_scores = cosine_similarity(query_embedding.reshape(1,-1), EMBEDDINGS)[0]
    
    similarity_indices = np.argsort(-similarity_scores)

    top_k_indices = similarity_indices[:top_k]

    return top_k_indices

import ipywidgets as widgets
from IPython.display import display, Markdown

def display_widget(llm_call_func):
    def on_button_click(b):
        # Clear outputs
        output1.clear_output()
        output2.clear_output()
        status_output.clear_output()
        # Display "Generating..." message
        status_output.append_stdout("Generating...\n")
        query = query_input.value
        top_k = slider.value
        prompt = prompt_input.value.strip() if prompt_input.value.strip() else None
        response1 = llm_call_func(query, use_rag=True, top_k=top_k, prompt=prompt)
        response2 = llm_call_func(query, use_rag=False, top_k=top_k, prompt=prompt)
        # Update responses
        with output1:
            display(Markdown(response1))
        with output2:
            display(Markdown(response2))
        # Clear "Generating..." message
        status_output.clear_output()

    query_input = widgets.Text(
        description='Query:',
        placeholder='Type your query here',
        layout=widgets.Layout(width='100%')
    )

    prompt_input = widgets.Textarea(
        description='Augmented prompt layout:',
        placeholder=("Type your prompt layout here, don't forget to add {query} and {documents} "
                     "where you want them to be placed! Leaving this blank will default to the "
                     "prompt in generate_final_prompt. Example:\nThis is a query: {query}\nThese are the documents: {documents}"),
        layout=widgets.Layout(width='100%', height='100px'),
        style={'description_width': 'initial'}
    )

    slider = widgets.IntSlider(
        value=5,  # default value
        min=1,
        max=20,
        step=1,
        description='Top K:',
        style={'description_width': 'initial'}
    )

    output1 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    output2 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    status_output = widgets.Output()

    submit_button = widgets.Button(
        description="Get Responses",
        style={'button_color': '#f0f0f0', 'font_color': 'black'}
    )
    submit_button.on_click(on_button_click)

    label1 = widgets.Label(value="With RAG", layout={'width': '45%', 'text_align': 'center'})
    label2 = widgets.Label(value="Without RAG", layout={'width': '45%', 'text_align': 'center'})

    display(widgets.HTML("""
    <style>
        .custom-output {
            background-color: #f9f9f9;
            color: black;
            border-radius: 5px;
        }
        .widget-textarea, .widget-button {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
        .widget-output {
            background-color: #f9f9f9 !important;
            color: black !important;
        }
        textarea {
            background-color: #fff !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
    </style>
    """))

    display(query_input, prompt_input, slider, submit_button, status_output)
    hbox_labels = widgets.HBox([label1, label2], layout={'justify_content': 'space-between'})
    hbox_outputs = widgets.HBox([output1, output2], layout={'justify_content': 'space-between'})

    def style_outputs(*outputs):
        for output in outputs:
            output.layout.margin = '5px'
            output.layout.height = '300px'
            output.layout.padding = '10px'
            output.layout.overflow = 'auto'
            output.add_class("custom-output")

    style_outputs(output1, output2)
    # Display label and output boxes
    display(hbox_labels)
    display(hbox_outputs)