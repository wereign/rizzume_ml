import json
from typing import List
from ollama import Client
from pydantic import BaseModel
import yaml
from profile_model import MasterProfile

PROMPT_PATH = './prompts.yaml'


class Tags(BaseModel):
    generated_tags: List[str]
    

class TagGenerator():
    def __init__(self):
        self.client = Client()
        self.model_name = "llama3.2"
        with open(PROMPT_PATH,'r') as pf:
            prompts = yaml.safe_load(pf)
        
        self.system_prompt = prompts['generate_tags']['system_prompt']
        self.user_prompt = prompts['generate_tags']['user_prompt']

    def generate_all_tags(self,master_profile:MasterProfile):
        
        mp_copy = (master_profile.model_copy().model_dump())
        del mp_copy['personal_info']

        print(mp_copy)

        messages = [
            {"role":"system","content":self.system_prompt},
            {"role":"user","content":self.user_prompt.format(resume=mp_copy)},
        ]

        response = self.client.chat(model=self.model_name,messages=messages,
                         format=Tags.model_json_schema())
        
        tags = json.loads(response.message.content)
        
        print(tags)

        return tags





if __name__ == "__main__":
    
    with open('./debug.json') as dj:
        payload = json.load(dj)
    master_profile = MasterProfile.model_validate(payload['master_profile'])

    tg = TagGenerator()
    tg.generate_all_tags(master_profile)