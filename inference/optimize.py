import yaml
import json
from ollama import ChatResponse, chat
from profile_model import MasterProfile

MODEL = 'llama3.2'

def load_job_description(path):
    with open(path) as file:
        jd = file.read()
        return jd

def load_json(file_path):
    """Load a JSON file into a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}")
        return None
    
def construct_messages(job_description_str,profile_json,prompts_path='./prompts.yaml'):
    with open(prompts_path) as sf:
        prompts = yaml.safe_load(sf)
        
    system_prompt = prompts['system_prompt']
    user_prompt = prompts['user_prompt'].format(profile=profile_json,jd=job_description_str)
    messages = [
        {"role":"system",
         "content":system_prompt},
        {"role":"user",
         "content":user_prompt}
    ]
    return messages


def optimize_profile(job_description:str,profile_json:dict,model:str=MODEL):
    """

    Args:
        job_description (str): _description_
        profile_json (dict): _description_
        model (str, optional): _description_. Defaults to MODEL.
    """

    messages = construct_messages(job_description,profile_json)
    response : ChatResponse = chat(model,messages=messages,format=MasterProfile.model_json_schema())
    response_txt = response.message.content
    
    return response_txt
    



if __name__ == "__main__":
    jd = load_job_description('./job_description.txt')
    pf = load_json('./aarav.json')

    print(optimize_resume(job_description=jd,profile_json=pf))