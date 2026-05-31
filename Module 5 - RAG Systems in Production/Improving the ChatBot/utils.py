# Importing necessary modules and libraries
from flask import Flask, request, jsonify
import ipywidgets as widgets
from IPython.display import display
import json
import logging
import os
import re
import requests
import threading
import markdown
from typing import List, Dict, Any, Union
from together import Together
from contextlib import contextmanager
import subprocess
from weaviate.classes.query import Filter
import joblib
import pandas as pd
import time
import httpx
from openai import OpenAI, DefaultHttpxClient
from opentelemetry.trace import Status, StatusCode
from sentence_transformers import SentenceTransformer

# Load a pretrained model from Hugging Face
model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder = ".models")

# Custom transport to bypass SSL verification
transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=False)

# Create a DefaultHttpxClient instance with the custom transport
http_client = DefaultHttpxClient(transport=transport)

def kill_processes_on_ports(
    ports: List[int],
    *,
    only_listening: bool = True,
    include_udp: bool = True,
    force: bool = True,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """
    Kill processes bound to any port in `ports`.

    Args:
        ports: List of port numbers to check and kill processes on
        only_listening: Kill only listeners (recommended)
        include_udp: Also consider UDP sockets
        force: Send SIGKILL/kill() if not gone after timeout
        timeout: Seconds to wait after terminate() before kill()

    Returns:
        Dict with details:
        {
            'pids_targeted': [int],
            'terminated': [{'pid': int, 'name': str}],
            'killed': [{'pid': int, 'name': str}],
            'errors': [{'pid': int, 'error': str}],
            'ports_with_no_match': [int]
        }

    Example:
        kill_processes_on_ports([5000, 8080, 3000])
    """
    import socket
    import psutil

    target_ports = {int(p) for p in ports}
    results = {
        'pids_targeted': [],
        'terminated': [],
        'killed': [],
        'errors': [],
        'ports_with_no_match': []
    }

    # Gather connections once (faster than per-process scanning)
    try:
        conns = psutil.net_connections(kind='inet')  # TCP + UDP IPv4/6
    except Exception as e:
        raise RuntimeError(f"Failed to enumerate network connections: {e}")

    # Map ports -> PIDs
    pids = set()
    matched_ports = set()
    for c in conns:
        if not c.laddr:
            continue
        port = c.laddr.port
        if port not in target_ports:
            continue

        # Filter protocol
        if c.type == socket.SOCK_STREAM:
            # TCP
            if only_listening and c.status != psutil.CONN_LISTEN:
                continue
        elif c.type == socket.SOCK_DGRAM:
            # UDP has no 'LISTEN' state; treat as listening if include_udp
            if not include_udp:
                continue

        if c.pid is not None:
            pids.add(c.pid)
            matched_ports.add(port)

    results['pids_targeted'] = sorted(pids)
    results['ports_with_no_match'] = sorted(target_ports - matched_ports)

    # Terminate, wait, then force-kill if requested
    procs = []
    for pid in list(pids):
        try:
            p = psutil.Process(pid)
            # Skip already-exiting processes
            if not p.is_running():
                continue
            p.terminate()
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as e:
            results['errors'].append({'pid': pid, 'error': f'Access denied: {e}'})
        except Exception as e:
            results['errors'].append({'pid': pid, 'error': str(e)})

    gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in gone:
        try:
            results['terminated'].append({'pid': p.pid, 'name': p.name()})
        except Exception:
            results['terminated'].append({'pid': p.pid, 'name': '?'})

    if alive and force:
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied as e:
                results['errors'].append({'pid': p.pid, 'error': f'Access denied on kill: {e}'})
            except Exception as e:
                results['errors'].append({'pid': p.pid, 'error': str(e)})
        gone2, alive2 = psutil.wait_procs(alive, timeout=timeout)
        for p in gone2:
            try:
                results['killed'].append({'pid': p.pid, 'name': p.name()})
            except Exception:
                results['killed'].append({'pid': p.pid, 'name': '?'})
        for p in alive2:
            results['errors'].append({'pid': p.pid, 'error': 'still alive after kill()'})

    return results

import os
def make_url(path = None):
    """
    Generate the Phoenix UI URL based on the platform.
    Supports three environments:
    - Coursera: Uses WORKSPACE_ID to build coursera.org URL (port 8888 with nginx proxy to Phoenix)
    - Learning Platform: Uses HOSTNAME and REV_PROXY_BASE_DOMAIN for custom domain (direct access to port 6006)
    - Local Machine: Falls back to localhost:6006 when no platform env vars are detected
    """
    BOLD = "\033[1m"
    RESET = "\033[0m"

    if path is None:
        path = "/"

    if 'WORKSPACE_ID' in os.environ:
        # Running on Coursera
        lab_id = os.environ['WORKSPACE_ID']
        url = f"http://{lab_id}.labs.coursera.org{path}"
    elif 'HOSTNAME' in os.environ and 'REV_PROXY_BASE_DOMAIN' in os.environ:
        # Running on Learning Platform
        ip = os.environ['HOSTNAME'].split('.')[0][3:]
        port = 6006  # Phoenix service port
        url = os.environ['REV_PROXY_BASE_DOMAIN'].format(ip=ip, port=port) + path
    else:
        # Running on local machine
        url = "http://localhost:6006" + path
        print(f"{BOLD}Running on local machine - using localhost{RESET}")

    print(f"{BOLD}FOLLOW THIS URL TO OPEN THE UI: {url}{RESET}")

def process_and_print_query(query, correct_label, response_std, tokens_std, response_simp, tokens_simp):
    # Color formatting for standard prompt results
    max_tokens = 180
    label_std_colored = (
        "\033[32m" + response_std + "\033[0m"
        if response_std == correct_label
        else "\033[31m" + response_std + "\033[0m"
    )
    tokens_std_colored = (
        "\033[32m" + str(tokens_std) + "\033[0m"
        if tokens_std <= 130
        else "\033[31m" + str(tokens_std) + "\033[0m"
    )
    
    # Color formatting for simplified prompt results
    label_simp_colored = (
        "\033[32m" + response_simp + "\033[0m"
        if response_simp == correct_label
        else "\033[31m" + response_simp + "\033[0m"
    )
    tokens_simp_colored = (
        "\033[32m" + str(tokens_simp) + "\033[0m"
        if tokens_simp <= max_tokens
        else "\033[31m" + str(tokens_simp) + "\033[0m"
    )

    # Print results
    print(f"Query: {query}")
    print(f"  Standard    → Label: {label_std_colored} | Tokens: {tokens_std_colored}")
    print(f"  Simplified  → Label: {label_simp_colored} | Tokens: {tokens_simp_colored}\n")


# Define utility functions and classes
def generate_with_single_input(prompt: str, role: str = 'user', top_p: float = None, 
                               temperature: float = None, max_tokens: int = 500, 
                               model: str = "Qwen/Qwen3.5-9B",
                               together_api_key=None, **kwargs):
    
    # Do not convert None to 'none' string here
    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
        **kwargs
    }
    
    # Only add them to the dict if they are actually set
    if top_p is not None:
        payload["top_p"] = top_p
    if temperature is not None:
        payload["temperature"] = temperature
    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        **kwargs
    }
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        url = os.path.join(os.environ.get("TOGETHER_BASE_URL", "https://proxy.dlai.link/coursera_proxy/together"), 'v1/chat/completions')
        response = requests.post(url, json=payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: f{response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key=together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()

    return json_dict

def generate_with_multiple_input(messages: List[Dict], top_p: float = 1, temperature: float = 1, max_tokens: int = 500,
                                 model: str = "Qwen/Qwen3.5-9B", together_api_key=None,
                                 **kwargs):
    payload = {
        "model": model,
        "messages": messages,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        client = OpenAI(
    api_key = '', # Set any as dlai proxy does not use it. Set the together api key if using the together endpoint
    base_url="http://proxy.dlai.link/coursera_proxy/together/", # If using together endpoint, add it here https://api.together.xyz/
   http_client=http_client, # ssl bypass to make it work via proxy calls, remove it if running with together.ai endpoint 
)
        try:
            json_dict = client.chat.completions.create(
                **payload,
                extra_body={"reasoning": False},
            ).model_dump()
            #json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].lower()
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key=together_api_key)
        json_dict = client.chat.completions.create(
            **payload,
            reasoning={"enabled": False},
        ).model_dump()
        #json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()
    try:
        output_dict = json_dict#{'role': json_dict['choices'][-1]['message']['role'],
                      # 'content': json_dict['choices'][-1]['message']['content'],
                      #'total_tokens':json_dict['usage']['total_tokens']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict

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

# Define utility functions and classes
def generate_embedding(prompt: str): #model: str = "BAAI/bge-base-en-v1.5", together_api_key = None, **kwargs):
    return model.encode(prompt).tolist()
    payload = {
        "model": model,
        "input": prompt,
        **kwargs
    }
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        client = OpenAI(
    api_key = '', # Set any as dlai proxy does not use it. Set the together api key if using the together endpoint
    base_url="http://proxy.dlai.link/coursera_proxy/together/", # If using together endpoint, add it here https://api.together.xyz/
   http_client=http_client, # ssl bypass to make it work via proxy calls, remove it if running with together.ai endpoint 
)
        try:
            json_dict = client.embeddings.create(**payload).model_dump()
            return json_dict['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key=together_api_key)
        try:
            json_dict = client.embeddings.create(**payload).model_dump()
            return json_dict['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}")

class ChatBot:
    """
    A simple chatbot class for handling user interactions using an LLM.
    
    This class maintains a conversation context and interfaces with a language model API 
    to generate responses related to a clothing store.
    """

    def __init__(self,  generator_function, tracer, model: str = "Qwen/Qwen3.5-9B", context_window: int = 20):
        # Initialize system and greeting messages
        self.system_prompt = {
            'role': 'system', 
            'content': "You are a friendly assistant from Fashion Forward Hub. It is a cloth store selling a variety of items. Your job is to answer questions related to FAQ or Products."
        }
        self.initial_message = {
            'role': 'assistant', 
            'content': "Hi! How can I help you?"
        }
        self.generator_function = generator_function
        self.tracer = tracer
        columns = ['query', 'result', 'total_tokens', 'kwargs']

        # Create an empty DataFrame with the specified columns
        self.logging_dataset = pd.DataFrame(columns=columns)
        

        # Initialize conversation with system and assistant message
        self.conversation: List[Dict[str, str]] = [self.system_prompt, self.initial_message]
        self.context_window = context_window  # Limit of past messages to consider
        self.model = model  # Model name to use for inference
        self.kwargs_list = []

    def call_llm(self, messages: List[Dict[str, str]], temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Calls the language model API with the provided messages and parameters.
        """
        try:
            response = together_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning={"enabled": False},
            )
            return response.model_dump()
        except Exception as e:
            print(f"[Error] LLM API call failed: {e}")
            return {"error": str(e)}


    def chat(self, prompt: str, role: str = 'user', return_stats = False) -> Dict[str, str]:
        """
        Handles a single round of user interaction and updates the conversation context.
        """
        with self.tracer.start_as_current_span("agent_call", openinference_span_kind="agent") as span: 
            span.set_input({"prompt":prompt, "role":role})
            start_time = time.time()
            recent_context = self.conversation[-self.context_window:]  # Get recent messages
            span.set_attribute("agent.recent_context", str(recent_context))
            params_dict, total_tokens = self.generator_function(prompt)  # Build API query parameters
            #params_dict['model'] = 'Qwen/Qwen3.5-9B'
            self.kwargs_list.append(params_dict)
            with self.tracer.start_as_current_span("llm_call", openinference_span_kind="llm") as llm_span:      
                try:
                    response = call_llm_with_context(context=recent_context, **params_dict)# Get response from model
                    llm_span.set_input({"messages":recent_context, **params_dict})
                    content = response['choices'][0]['message']['content']
                    total_tokens = response['usage']['total_tokens']
                except Exception as error:
                    llm_span.record_exception(error)
                    llm_span.set_status(Status(StatusCode.ERROR))
                    raise
                else:
                    # OpenInference Semantic Conventions for computing Costs
                    llm_span.set_attribute("llm.token_count.prompt", response['usage']['prompt_tokens'])
                    llm_span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
                    llm_span.set_attribute("llm.token_count.total", response['usage']['total_tokens'])
                    llm_span.set_attribute("llm.model_name", response['model'])
                    llm_span.set_attribute("llm.provider", 'together.ai')
                    llm_span.set_output(response)
                    llm_span.set_status(Status(StatusCode.OK))
                    
                self.conversation.append({"role": "user", 'content': prompt})  # Append user message
                self.conversation.append({"role":"assistant", "content":content})  # Append latest assistant message
                #self.logging_function(prompt, params_dict, total_tokens, content, logging_dataset = self.logging_dataset)
                end_time = time.time()
                total_time = end_time - start_time
                if return_stats:
                    return content, total_tokens, total_time
                span.set_output({"content": content, 'total_tokens':total_tokens})
                span.set_status(Status(StatusCode.OK))   
                return {"content":content, "role":role}

    def start_conversation(self) -> None:
        """
        Starts a terminal-based conversation loop with the user.
        """
        print(self.format_message(self.initial_message))
        while True:
            prompt = input("You: ")
            if prompt.lower() == 'end conversation':
                break
            print(f"User: {prompt}")
            response = self.chat(prompt)
            print(self.format_message(response))

    def clear_conversation(self) -> None:
        """
        Resets the conversation to its initial state.
        """
        self.conversation = [self.system_prompt, self.initial_message]


class ChatWidget:
    """
    A widget-based UI for interacting with the ChatBot using ipywidgets.
    
    Displays messages, handles user input, and dynamically loads related images.
    """

    def __init__(self, generator_function, tracer):
        self.chat_bot = ChatBot(generator_function, tracer)  # Initialize chatbot
        self.output_area = widgets.HTML()  # Area to show conversation text
        self.image_area = widgets.HBox()  # Area to display images
        self.text_input = widgets.Text(placeholder='Type your message...', layout=widgets.Layout(width='90%'))
        self.send_button = widgets.Button(description='Send', layout=widgets.Layout(width='10%'))

        # Attach click event to send button
        self.send_button.on_click(self.send_message)

        self.unique_ids = set()  # Track which image IDs were already shown

        self.display()  # Display the UI
        self.refresh_messages()  # Show initial messages

    def send_message(self, _):
        """
        Handles sending a user message from the text input field.
        """
        user_message = self.text_input.value
        if user_message.strip() == "":
            return

        self.display_user_message(user_message)  # Show user message
        self.show_thinking()  # Show 'thinking...' message
        self.text_input.value = ''  # Clear input box
        self.image_area.children = ()  # Reset image display

        # Handle bot response in a separate thread
        threading.Thread(target=self.process_bot_response, args=(user_message,)).start()

    def process_bot_response(self, user_message):
        """
        Gets the response from the bot and updates the UI accordingly.
        """
        
        response = self.chat_bot.chat(user_message)
        response_content = response['content']
        self.extract_and_process_ids(response_content)  # Process any mentioned IDs
        self.refresh_messages()  # Refresh message display

    def extract_and_process_ids(self, message: str):
        """
        Finds and processes any 'ID: <number>' entries in the message to load images.
        """
        pattern = re.compile(r'ID:\s*(\d+(?:,\s*\d+)*)', re.IGNORECASE)
        matches = pattern.findall(message)
        found_ids = [id.strip() for match in matches for id in match.split(',')]
        for id in found_ids:
            if id not in self.unique_ids:
                self.unique_ids.add(id)
                self.load_image(id)

    def load_image(self, id: str):
        """
        Loads an image associated with the given ID and displays it in the image area.
        """
        image_path = f"/home/jovyan/data/collections/collections_assignment_4/images/{id}.jpg"
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
            img_widget = widgets.Image(value=img_data, format='jpg',
                                       layout=widgets.Layout(width='150px', height='auto', margin='5px'))
            id_label = widgets.Label(value=f"ID: {id}", layout=widgets.Layout(width='150px'))
            vbox = widgets.VBox([img_widget, id_label])
            self.image_area.children += (vbox,)

    def display_user_message(self, message: str):
        """
        Displays the user's message in the output HTML area.
        """
        escaped_message = markdown.markdown(message)
        html_content = self.output_area.value
        html_content += (f"<div style='background-color: #f1f1f1; padding: 10px; margin: 5px 0; "
                         f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>User:</strong>"
                         f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_message}</div></div>")
        self.output_area.value = html_content

    def show_thinking(self):
        """
        Temporarily shows a 'Thinking...' placeholder from the assistant.
        """
        html_content = self.output_area.value
        html_content += (f"<div style='background-color: #fff3cd; padding: 10px; margin: 5px 0; "
                         f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>Assistant:</strong>"
                         f"<div style='margin: 0; white-space: normal; color: black !important;'>Thinking...</div></div>")
        self.output_area.value = html_content

    def refresh_messages(self):
        """
        Re-renders the entire conversation history in the output area.
        """
        html_content = "<div style='font-family: Arial; max-width: 600px;'>"
        
        for message in self.chat_bot.conversation:
            if message['role'] == 'user':
                escaped_content = markdown.markdown(message['content'])
                html_content += (f"<div style='background-color: #f1f1f1; padding: 10px; margin: 5px 0; "
                                 f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>User:</strong>"
                                 f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_content}</div></div>")
            elif message['role'] == 'assistant':
                formatted_message = message['content']
                escaped_content = markdown.markdown(formatted_message)
                html_content += (f"<div style='background-color: #e2f7d5; padding: 10px; margin: 5px 0; "
                                 f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>Assistant:</strong>"
                                 f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_content}</div></div>")
        html_content += "</div>"
        self.output_area.value = html_content

    def display(self):
        """
        Sets up and displays the chat widget user interface with images aligned at the bottom right.
        """
        # Set up the input area with text input and send button.
        input_area = widgets.HBox([self.text_input, self.send_button])
        chat_ui = widgets.VBox([self.output_area, self.image_area, input_area],
                               layout=widgets.Layout(margin='10px'))
        display(chat_ui)

def print_object_properties(obj: Union[dict, list]) -> None:
    t = ''
    if isinstance(obj, dict):
        for x, y in obj.items():
            if x == 'article_content':
                t += f'{x}: {y[:100]}...(truncated)\n'
            elif x == 'main_vector':
                t += f'{x}: {y[:30]}...(truncated)\n'
            elif x == 'chunk':
                t += f'{x}: {y[:100]}...(truncated)\n'
            else:
                t += f'{x}: {y}\n'
    else:
        for l in obj:
            for x, y in l.items():
                if x == 'article_content':
                    t += f'{x}: {y[:100]}...(truncated)\n'
                elif x == 'main_vector':
                    t += f'{x}: {y[:30]}...(truncated)\n'
                elif x == 'chunk':
                    t += f'{x}: {y[:100]}...(truncated)\n'
                else:
                    t += f'{x}: {y}\n'
            t += "\n\n"
    print(t)

def call_llm_with_context(prompt: str, context: list, role: str = 'user', **kwargs):
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
    context.append({'role': role, 'content': prompt})
    response = generate_with_multiple_input(context, **kwargs)
    context.append(response)
    return response

def print_properties(item):
    print(
        json.dumps(
            item.properties,
            indent=2, sort_keys=True, default=str
        )
    )
    
def parse_json_output(llm_output):
    """
    Parses a string output from a language model into a JSON object.

    This function attempts to clean and parse a JSON-formatted string produced by a language model (LLM).
    The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
    instead of double quotes. The function attempts to correct such issues before parsing.

    Parameters:
    - llm_output (str): The string output from the language model that is expected to be in JSON format.

    Returns:
    - dict or None: A dictionary if parsing is successful, or None if the input string cannot be parsed into valid JSON.

    Exception Handling:
    - In case of a JSONDecodeError during parsing, an error message is printed, and the function returns None.
    """
    try:
        # Since the input might be improperly formatted, ensure any single quotes are removed
        llm_output = llm_output.replace("\n", '').replace("'",'').replace("}}", "}").replace("{{", "{")  # Remove any erroneous structures
        
        # Attempt to parse JSON directly provided it is a properly-structured JSON string
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None

def get_filter_by_metadata(json_output: dict | None = None):
    """
    Generate a list of Weaviate filters based on a provided metadata dictionary.

    Parameters:
    - json_output (dict) or None: Dictionary containing metadata keys and their values.

    Returns:
    - list[Filter] or None: A list of Weaviate filters, or None if input is None.
    """
    # If the input dictionary is None, return None immediately
    if json_output is None:
        return None

    # Define a tuple of valid keys that are allowed for filtering
    valid_keys = (
        'gender',
        'masterCategory',
        'articleType',
        'baseColour',
        'price',
        'usage',
        'season',
    )

    # Initialize an empty list to store the filters
    filters = []

    # Iterate over each key-value pair in the input dictionary
    for key, value in json_output.items():
        # Skip the key if it is not in the list of valid keys
        if key not in valid_keys:
            continue

        # Special handling for the 'price' key
        if key == 'price':
            # Ensure the value associated with 'price' is a dictionary
            if not isinstance(value, dict):
                continue

            # Extract the minimum and maximum prices from the dictionary
            min_price = value.get('min')
            max_price = value.get('max')

            # Skip if either min_price or max_price is not provided
            if min_price is None or max_price is None:
                continue

            # Skip if min_price is non-positive or max_price is infinity
            if min_price <= 0 or max_price == 'inf':
                continue

            # Add filters for price greater than min_price and less than max_price
            filters.append(Filter.by_property(key).greater_than(min_price))
            filters.append(Filter.by_property(key).less_than(max_price))
        else:
            # For other valid keys, add a filter that checks for any of the provided values
            filters.append(Filter.by_property(key).contains_any(value))

    return filters



def generate_filters_from_query(query):
    json_string, total_tokens = generate_metadata_from_query(query)
    json_output = parse_json_output(json_string)
    filters = get_filter_by_metadata(json_output)
    return filters, total_tokens

products_data = joblib.load('dataset/clothes_json.joblib')


# Run this cell to generate the dictionary with the possible values for each key
values = {}
for d in products_data:
    for key, val in d.items():
        if key in ('product_id', 'price', 'productDisplayName', 'subCategory', 'year'):
            continue
        if key not in values.keys():
            values[key] = set()
        values[key].add(val)

def generate_metadata_from_query(query):
    """
    Generates metadata in JSON format based on a given query to filter clothing items.

    This function constructs a prompt for a language model to create a JSON object that will
    guide the filtering of a vector database query for clothing items. It takes possible values from
    a predefined set and ensures only relevant metadata is included in the output JSON.

    Parameters:
    - query (str): The query describing specific clothing-related needs.

    Returns:
    - str: A JSON string representing metadata with keys like gender, masterCategory, articleType,
      baseColour, price, usage, and season. Each value in the JSON is within a list, with prices specified
      as a dict containing "min" and "max" values. Unrestricted keys should use ["Any"] and unspecified
      prices should default to {"min": 0, "max": "inf"}.
    """

    # Set the prompt. Remember to include the query, the desired JSON format, the possible values (passing {values} at some point) 
    # and explain to the LLM what is going on. 
    # Explicitly tell the llm to include gender, masterCategory, ArticleType, baseColour, price, usage and season as keys.
    # Also mention to the llm that price key must be a json with "min" and "max" values (0 if no lower bound and inf if no upper bound)
    # If there is no price set, add min = 0 and max = inf.
    PROMPT = f"""
    One query will be provided. For the given query, there will be a call on vector database to query relevant cloth items. 
    Generate a JSON with useful metadata to filter the products in the query. Possible values for each feature is in the following json: {values}

    Provide a JSON with the features that best fit in the query (can be more than one, write in a list). Also, if present, add a price key, saying if there is a price range (between values, greater than or smaller than some value).
    Only return the JSON, nothing more. price key must be a json with "min" and "max" values (0 if no lower bound and inf if no upper bound). 
    Always include gender, masterCategory, ArticleType, baseColour, price, usage and season as keys. All values must be within lists.
    If there is no price set, add min = 0 and max = inf.
    Only include values that are given in the json above. 
    
    Example of expected JSON:

    {{
    "gender": ["Women"],
    "masterCategory": ["Apparel"],
    "articleType": ["Dresses"],
    "baseColour": ["Blue"],
    "price": {{"min": 0, "max": "inf"}},
    "usage": ["Formal"],
    "season": ["All seasons"]
    }}

    Query: {query}
             """

    # Generate the response with the generate_with_single_input, PROMPT, temperature = 0 (low randomness) and max_tokens = 1500.
    response = generate_with_single_input(PROMPT, temperature = 0, max_tokens = 1500) # @REPLACE EQUALS None

    # Get the content
    content = response['content']
    
    total_tokens = response['total_tokens']

    
    return content, total_tokens
