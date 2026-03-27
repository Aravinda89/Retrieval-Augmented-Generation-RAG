import json
import requests
import os
import re
from typing import List, Dict, Any
from together import Together


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

def generate_params_dict(
    prompt: str,
    temperature: float = None,
    role = 'user',
    top_p: float = None,
    max_tokens: int = 500,
    model: str = "Qwen/Qwen3.5-9B"
):
    """
    Call an LLM with different sampling parameters to observe their effects.

    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (lower = more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        model: The model to use

    Returns:
        The LLM response
    """

    # Create the dictionary with the necessary parameters
    kwargs = {"prompt": prompt, 'role':role, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, 'model': model}


    return kwargs

def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="Qwen/Qwen3.5-9B",
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
        "reasoning": {"enabled": False},
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


def generate_with_multiple_input(messages: List[Dict],
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="Qwen/Qwen3.5-9B",
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
        "reasoning": {"enabled": False},
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

def call_llm_with_context(prompt: str, context: list,  role: str = 'user', **kwargs):
    """
    Calls a language model with the given prompt and context to generate a response.

    Parameters:
    - prompt (str): The input text prompt provided by the user.
    - role (str): The role of the participant in the conversation, e.g., "user" or "assistant".
    - context (list): A list representing the conversation history, to which the new input is added.
    - **kwargs: Additional keyword arguments for configuring the language model call (e.g., top_k, temperature).

    Returns:
    - response (str): The generated response from the language model based on the provided prompt and context.
    """

    # Append the dictionary {'role': role, 'content': prompt} into the context list
    context.append({'role': role, 'content': prompt})

    # Call the llm with multiple input passing the context list and the **kwargs
    response = generate_with_multiple_input(context, **kwargs)

    # Append the LLM response in the context dict
    context.append(response)

    return response
