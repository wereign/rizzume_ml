import yaml
import json
import ollama  # Ollama Python client
from profile_model import MasterProfile

MODEL = 'smollm2'

def load_job_description(path):
    """Load job description from a text file."""
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def load_json(file_path):
    """Load a JSON file into a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}")
        return None
    
def construct_messages(job_description_str, profile_json, prompts_path='./inference/prompts.yaml'):
    """Constructs messages for Ollama chat."""
    with open(prompts_path, 'r', encoding='utf-8') as sf:
        prompts = yaml.safe_load(sf)
        
    system_prompt = prompts['system_prompt']
    user_prompt = prompts['user_prompt'].format(profile=profile_json, jd=job_description_str)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def optimize_profile(job_description: str, profile_json: dict, model: str = MODEL):
    """
    Optimizes the profile based on the job description using Ollama.

    Args:
        job_description (str): The job description text.
        profile_json (dict): The profile JSON object.
        model (str, optional): The Ollama model to use. Defaults to MODEL.

    Returns:
        str: The response from Ollama.
    """

    messages = construct_messages(job_description, profile_json)
    
    # Create an Ollama client
    client = ollama.Client(host="http://ollama:11434")
    
    # Send the chat request to Ollama
    response = client.chat(model=model, messages=messages)
    
    # Extract response text
    response_txt = response.get('message', {}).get('content', 'No response received.')
    
    return response_txt


if __name__ == "__main__":
    jd = load_job_description('./job_description.txt')
    pf = load_json('./aarav.json')

    print(optimize_profile(job_description=jd, profile_json=pf))
