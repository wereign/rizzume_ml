from pprint import pprint
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
    

def filter_selected_sections(profile, selected_tags):
    filtered_profile = {}

    for section, items in profile.items():
        # Keep personal_info unchanged
        if section == "personal_info":
            filtered_profile[section] = items
            continue
        
        dropped_count = {}
        # Ensure we only process lists (e.g., education, skills, etc.)
        if isinstance(items, list):
            filtered_section = []
             
            for item in items:
                # If 'tags' exist and at least one matches, keep the item
                if 'tags' in item and any(tag in selected_tags for tag in item['tags']):
                    filtered_section.append(item)
                

                # If 'tags' is absent, keep the item by default
                elif 'tags' not in item:
                    filtered_section.append(item)
                else:
                    if section in dropped_count:
                        dropped_count[section] += 1
                    else:
                        dropped_count[section] = 1

            # Only add the section if there are matching items
            if filtered_section:
                filtered_profile[section] = filtered_section
        print("Dropped Items Count",dropped_count)
    return filtered_profile
    
def construct_messages(job_description_str, profile, prompts_path='./inference/prompts.yaml'):
    """Constructs messages for Ollama chat."""
    with open(prompts_path, 'r', encoding='utf-8') as sf:
        prompts = yaml.safe_load(sf)
        
    system_prompt = prompts['system_prompt']
    user_prompt = prompts['user_prompt'].format(profile=profile, jd=job_description_str)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def optimize_profile(job_description: str, profile: MasterProfile, selected_tags:list, model: str = MODEL):
    """
    Optimizes the profile based on the job description using Ollama.

    Args:
        job_description (str): The job description text.
        profile_json (dict): The profile JSON object.
        model (str, optional): The Ollama model to use. Defaults to MODEL.

    Returns:
        str: The response from Ollama.
    """
    profile = json.loads(profile.model_dump_json())
    filtered_profile = filter_selected_sections(profile,selected_tags)
    messages = construct_messages(job_description, filtered_profile)
    
    # Create an Ollama client
    client = ollama.Client(host="http://localhost:11434")
    model_list = client.list()
    client.pull(model=MODEL)
    print()
    print(model_list,type(model_list))
    
    # Send the chat request to Ollama
    response = client.chat(model=model, messages=messages,format=MasterProfile.model_json_schema())
    
    # Extract response text
    response_txt = response.get('message', {}).get('content', 'No response received.')
    
    return response_txt


if __name__ == "__main__":
    jd = load_job_description('./job_description.txt')
    pf = load_json('./aarav.json')

    print(optimize_profile(job_description=jd, profile_json=pf))
