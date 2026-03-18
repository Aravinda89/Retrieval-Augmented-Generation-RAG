import json
import requests
from typing import Union, List, Dict, Any
import os
from contextlib import redirect_stdout, redirect_stderr
import logging
import httpx
from openai import OpenAI, DefaultHttpxClient

from sentence_transformers import SentenceTransformer

# Load a pretrained model from Hugging Face
model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder = ".models")

# Custom transport to bypass SSL verification
transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=False)

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

def print_object_properties(obj: Union[dict, list]) -> None:
    t = ''
    if isinstance(obj, dict):
        keys = list(obj.keys())
        keys.sort()
        for x in keys:
            y = obj[x]
            if x == 'article_content':
                t += f'{x}: {y[:100]}...(truncated)\n'
            elif x == 'main_vector':
                t+= f'{x}: {y[:30]}...(truncated)\n'
            elif x == 'chunk':
                t+= f'{x}: {y[:100]}...(truncated)\n'

            else:
                t+= f'{x}: {y}\n'
    else:
        for l in obj:
            print_object_properties(l)
        
    print(t)
    
    

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


import json
from together import Together
import os
# Define utility functions and classes
def generate_with_single_input(prompt: str, role: str = 'user', top_p: float = None, 
                               temperature: float = None, max_tokens: int = 500, 
                               model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'],
                       'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict

def generate_with_multiple_input(messages: List[Dict], 
                               top_p: float = 1, 
                               temperature: float = 1,
                               max_tokens: int = 500,
                               model: str ="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
                                together_api_key = None,
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
        client = Together(api_key =  together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict




def print_properties(item):
    print(
        json.dumps(
            item.properties,
            indent=2, sort_keys=True, default=str
        )
    )


import ipywidgets as widgets
from IPython.display import display, Markdown

def display_widget(llm_call_func, semantic_search_retrieve, bm25_retrieve, hybrid_retrieve, semantic_search_with_reranking):
    def on_button_click(b):
        query = query_input.value
        top_k = slider.value
        rerank_property = rerank_property_dropdown.value
        # Clear existing outputs
        for output in [output1, output2, output3, output4, output5]:
            output.clear_output()
        status_output.clear_output()
        # Display "Generating..." message
        status_output.append_stdout("Generating...\n")
        # Update outputs one by one
        retrievals = [
            (output1, llm_call_func, query, semantic_search_retrieve, True),
            (output4, llm_call_func, query, semantic_search_with_reranking, True, True, rerank_property),
            (output2, llm_call_func, query, bm25_retrieve, True),
            (output3, llm_call_func, query, hybrid_retrieve, True),
            (output5, llm_call_func, query, None, False)  # Without RAG
        ]
        for output, func, query, retriever, use_rag, *extra_info in retrievals:
            use_rerank = extra_info[0] if len(extra_info) > 0 else False
            if use_rerank:
                rr_property = extra_info[1]
            else:
                rr_property = None
            response = func(query=query, top_k=top_k, use_rag=use_rag, retrieve_function=retriever, use_rerank=use_rerank, rerank_property=rr_property)
            with output:
                display(Markdown(response))
        # Clear "Generating..." message
        status_output.clear_output()
    
    query_input = widgets.Text(
        description='',
        value="Tell me about United States and Brazil's relationship over the course of 2024. Provide links for the resources you use in the answer.",
        placeholder='Type your query here',
        layout=widgets.Layout(width='70%')  # Adjusted width to make room for label
    )
    
    query_label = widgets.Label(value="Query:", layout=widgets.Layout(width='10%', align='center'))
    query_box = widgets.HBox([query_label, query_input], layout=widgets.Layout(align_items='center'))  # Positioned side by side

    slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Top K:',
        style={'description_width': 'initial'}
    )
    
    rerank_property_dropdown = widgets.Dropdown(
        options=['title', 'chunk'],
        value='title',
        description='Rerank Property:',
        style={'description_width': 'initial'}
    )
    
    output_style = {'border': '1px solid #ccc', 'width': '100%'}
    output1 = widgets.Output(layout=output_style)
    output2 = widgets.Output(layout=output_style)
    output3 = widgets.Output(layout=output_style)
    output4 = widgets.Output(layout=output_style)
    output5 = widgets.Output(layout=output_style)
    status_output = widgets.Output()
    
    submit_button = widgets.Button(
        description="Get Responses",
        style={'button_color': '#eee', 'font_color': 'black'}
    )
    submit_button.on_click(on_button_click)
    
    label1 = widgets.Label(value="Semantic Search")
    label2 = widgets.Label(value="BM25 Search")
    label3 = widgets.Label(value="Hybrid Search")
    label4 = widgets.Label(value="Semantic Search with Reranking")
    label5 = widgets.Label(value="Without RAG")
    
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
    
    display(query_box, slider, rerank_property_dropdown, submit_button, status_output)
    
    # Create individual vertical containers for each label and output
    vbox1 = widgets.VBox([label1, output1], layout={'width': '30%'})
    vbox2 = widgets.VBox([label2, output2], layout={'width': '30%'})
    vbox3 = widgets.VBox([label3, output3], layout={'width': '30%'})
    vbox4 = widgets.VBox([label4, output4], layout={'width': '30%'})
    vbox5 = widgets.VBox([label5, output5], layout={'width': '30%'})
    
    # HBoxes to arrange VBoxes in rows
    hbox_outputs1 = widgets.HBox([vbox1, vbox4, vbox2], layout={'justify_content': 'space-between'})
    hbox_outputs2 = widgets.HBox([vbox3, vbox5], layout={'justify_content': 'space-between'})
    
    def style_outputs(*outputs):
        for output in outputs:
            output.layout.margin = '5px'
            output.layout.height = '300px'
            output.layout.padding = '10px'
            output.layout.overflow = 'auto'
            output.add_class("custom-output")
            
    style_outputs(output1, output2, output3, output4, output5)
    
    # Display rows with outputs
    display(hbox_outputs1)
    display(hbox_outputs2)