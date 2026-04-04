# Importing necessary modules and libraries
from flask import Flask, request, jsonify
import json
import os
import requests
from contextlib import contextmanager
import subprocess
import joblib
import time
from together import Together
import weaviate
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
# Load a pretrained model from Hugging Face
model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder = ".models")


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

def make_url():
    """
    Generate the Phoenix UI URL based on the platform.
    Supports three environments:
    - Coursera: Uses WORKSPACE_ID to build coursera.org URL (port 8888 with nginx proxy to Phoenix)
    - Learning Platform: Uses HOSTNAME and REV_PROXY_BASE_DOMAIN for custom domain (direct access to port 6006)
    - Local Machine: Falls back to localhost:6006 when no platform env vars are detected
    """
    BOLD = "\033[1m"
    RESET = "\033[0m"

    if 'WORKSPACE_ID' in os.environ:
        # Running on Coursera
        lab_id = os.environ['WORKSPACE_ID']
        url = f"http://{lab_id}.labs.coursera.org"
    elif 'HOSTNAME' in os.environ and 'REV_PROXY_BASE_DOMAIN' in os.environ:
        # Running on Learning Platform
        ip = os.environ['HOSTNAME'].split('.')[0][3:]
        port = 6006  # Phoenix service port
        url = os.environ['REV_PROXY_BASE_DOMAIN'].format(ip=ip, port=port)
    else:
        # Running on local machine
        url = "http://localhost:6006"
        print(f"{BOLD}Running on local machine - using localhost{RESET}")

    print(f"{BOLD}FOLLOW THIS URL TO OPEN THE UI: {url}{RESET}")

def restart_kernel():
    # This forces the kernel to restart by exiting Python
    import os
    os._exit(00)  # Exiting the Python process itself

# Define utility functions and classes
def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str = "Qwen/Qwen3.5-9B",
                               together_api_key=None,
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
        response = requests.post(url, json=payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key=together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'],
                       'content': json_dict['choices'][-1]['message']['content'],
                      'total_tokens':json_dict['usage']['total_tokens']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict


# Define utility functions and classes
def generate_embedding(prompt: str): #model: str = "BAAI/bge-base-en-v1.5", together_api_key = None, **kwargs):
    return model.encode(prompt).tolist()
    payload = {
        "model": model,
        "input": prompt,
        **kwargs
    }
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        url = os.path.join(get_proxy_url(), 'v1/embeddings')
        response = requests.post(url, json=payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
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
        


def cleanup_phoenix_projects():
    """
    Cleanup function to resolve Phoenix project ID conflicts.
    Creates fresh Phoenix session with unique project name.
    """
    try:
        import phoenix as px
        import time
        # Stop any existing Phoenix sessions
        px.close_app()
        time.sleep(2)  # Brief pause for cleanup
        print("Phoenix cleanup completed")
    except Exception as e:
        print(f"Phoenix cleanup warning: {e}")

def setup_faq_collection():
    """
    Sets up the Weaviate FAQ collection with data from faq.joblib.
    This function should be called to ensure the collection exists before using it.
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(port=8079, grpc_port=50050)
        
        # Check if collection already exists
        if client.collections.exists("Faq"):
            print("FAQ collection already exists")
            client.close()
            return
        
        # Create the collection with proper schema
        collection = client.collections.create(
            name="Faq",
            properties=[
                weaviate.classes.config.Property(
                    name="question",
                    data_type=weaviate.classes.config.DataType.TEXT,
                ),
                weaviate.classes.config.Property(
                    name="answer", 
                    data_type=weaviate.classes.config.DataType.TEXT,
                ),
                weaviate.classes.config.Property(
                    name="type",
                    data_type=weaviate.classes.config.DataType.TEXT,
                ),
            ],
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers()
        )
        
        # Load FAQ data from joblib file
        import os
        faq_file_path = os.path.join(os.path.dirname(__file__), "faq.joblib")
        faq_data = joblib.load(faq_file_path)
        
        # Insert data into collection
        collection.data.insert_many(faq_data)
        
        print(f"Successfully created FAQ collection with {len(faq_data)} items")
        client.close()
        
    except Exception as e:
        print(f"Error setting up FAQ collection: {e}")
        if 'client' in locals():
            client.close()
        raise
