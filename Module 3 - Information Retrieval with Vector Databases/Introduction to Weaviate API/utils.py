from flask import Flask, request, jsonify
import threading
from sentence_transformers import SentenceTransformer
import json
import requests
from typing import Union
import os
from contextlib import redirect_stdout, redirect_stderr
import json
import requests
from typing import Union, List, Dict, Any
import os
from contextlib import redirect_stdout, redirect_stderr
import logging
import logging
from typing import Dict, List
import together
import torch
import os
import subprocess
import signal
import sys
import logging
import httpx
from openai import OpenAI, DefaultHttpxClient

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


def generate_with_single_input(prompt: str, 
                               role: str = 'user', 
                               top_p: float = None, 
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
        url = os.path.join('https://proxy.dlai.link/coursera_proxy/together', 'v1/chat/completions')   
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
        url = os.path.join('https://proxy.dlai.link/coursera_proxy/together', 'v1/chat/completions')   
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




def print_object_properties(obj: Union[dict, list]) -> None:
    t = ''
    if isinstance(obj, dict):
        for x,y in obj.items():
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
            for x,y in l.items():
                if x == 'article_content':
                    t += f'{x}: {y[:100]}...(truncated)\n'
                elif x == 'main_vector':
                    t+= f'{x}: {y[:30]}...(truncated)\n'
                elif x == 'chunk':
                    t+= f'{x}: {y[:100]}...(truncated)\n'

                else:
                    t+= f'{x}: {y}\n'
            t += "\n\n"
        
    print(t)


def print_properties(item):
    print(
        json.dumps(
            item.properties,
            indent=2, sort_keys=True, default=str
        )
    )

import subprocess
from contextlib import contextmanager

@contextmanager
def suppress_subprocess_output():
    """
    Context manager that suppresses the standard output and error 
    of any subprocess.Popen calls within this context.
    """
    # Store the original Popen
    original_popen = subprocess.Popen

    def patched_popen(*args, **kwargs):
        # Redirect the stdout and stderr to subprocess.DEVNULL
        kwargs['stdout'] = subprocess.DEVNULL
        kwargs['stderr'] = subprocess.DEVNULL
        return original_popen(*args, **kwargs)

    try:
        # Apply the patch by replacing subprocess.Popen with patched_popen
        subprocess.Popen = patched_popen
        # Yield control back to the context
        yield
    finally:
        # Ensure that the original Popen method is restored
        subprocess.Popen = original_popen
