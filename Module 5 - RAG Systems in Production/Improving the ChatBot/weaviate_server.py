import weaviate
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
with suppress_subprocess_output():
    client = weaviate.connect_to_embedded(
        persistence_data_path="/home/jovyan/data/collections/collections_assignment_4/",
        #version="1.28.3",
        environment_variables={
            "ENABLE_API_BASED_MODULES": "true",
            "ENABLE_MODULES": 'text2vec-transformers',
            "TRANSFORMERS_INFERENCE_API":"http://127.0.0.1:5000/"
        }
    )


