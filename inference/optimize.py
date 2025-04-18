from pprint import pprint
from typing import List
import yaml
import json
import ollama
from copy import deepcopy
from profile_model import MasterProfile,  Project, Experience
from pydantic import BaseModel

ALL_MODELS = ['llama3.2', 'smollm2', 'gemma3:4b', 'gemma3:1b', 'command-r7b']

class ModelOutput(BaseModel):
    description : str
    relevance : int

class OptimizeResume():
    def __init__(self, llm_model, master_profile: MasterProfile, job_description: str, selected_tags: list, prompts_path: str = './prompts.yaml'):
        self.client = ollama.Client()
        if llm_model in ALL_MODELS:
            self.model_name = llm_model

        with open(prompts_path, 'r', encoding='utf-8') as p_file:
            self.prompts = yaml.safe_load(p_file)

        self.job_description = job_description
        self.master_profile = MasterProfile.model_validate(master_profile)
        self.selected_tags = selected_tags

        self.filtered_profile = None
        self.tuned_profile = None

    def filter_sections(self):

        print("Filtering Sections")
        def has_matching_tag(tags: List[str]) -> bool:
            return any(tag in selected_tags for tag in tags)

        filtered_profile = deepcopy(self.master_profile)
        master_profile = self.master_profile
        selected_tags = self.selected_tags
        
        filtered_profile.projects = [
            proj for proj in master_profile.projects if has_matching_tag(proj.tags)]
        filtered_profile.experience = [
            exp for exp in master_profile.experience if has_matching_tag(exp.tags)]

        if master_profile.certifications:
            filtered_profile.certifications = [
                cert for cert in master_profile.certifications if has_matching_tag(cert.tags)]
            if not filtered_profile.certifications:
                filtered_profile.certifications = None

        if master_profile.achievements:
            filtered_profile.achievements = [
                ach for ach in master_profile.achievements if has_matching_tag(ach.tags)]
            if not filtered_profile.achievements:
                filtered_profile.achievements = None

        if master_profile.publications:
            filtered_profile.publications = [
                pub for pub in master_profile.publications if has_matching_tag(pub.tags)]
            if not filtered_profile.publications:
                filtered_profile.publications = None

        self.filtered_profile = filtered_profile
        return filtered_profile

    def optimize_item(self, section_prompt, section_description):
        print("Optimizing item")
        item_message = section_prompt['user_prompt'].format(jd=self.job_description,item=section_description)
        messages = [
            {
                "role": "system", "content": section_prompt['system_prompt']
            },
            {
                "role": "user", "content": item_message 
            }
        ]

        response = self.client.chat(model=self.model_name,messages=messages,format=ModelOutput.model_json_schema())
        return json.loads(response.message.content)

    def tune_resume(self,as_json=True):
        print("Tuning Resumes")
        if not self.filtered_profile:
            self.filter_sections()
        else:
            pass
        
        tuned_profile = deepcopy(self.filtered_profile)
        # mutable sections - Project | Experience

        tune_dicts = [
            {"section": tuned_profile.projects,
                "prompts": self.prompts['projects']},
            {"section": tuned_profile.experience,
                "prompts": self.prompts['experience']}
        ]

        # optimize the mutable sections
        for td in tune_dicts:
            section = td['section']
            prompts = td['prompts']

            for item in section:
                print('-'*50)
                print(item)
                tuned_description = self.optimize_item(
                    prompts, item.description)
                item.description = tuned_description['description']
                item.relevance = tuned_description['relevance']

            print("original section")
            print(section)    
            section.sort(key=lambda x:x.relevance,reverse=True)
            print("Sorted section")
            print(section)    
        
            
        self.tuned_profile = tuned_profile

        if as_json:
            return json.loads(tuned_profile.model_dump_json())
        else:
            self.tuned_profile
        


if __name__ == "__main__":
    payload_path = './debug.json'
    with open(payload_path, 'r') as rf:
        payload = json.load(rf)

    master_profile = MasterProfile.model_validate_json(
        json.dumps(payload['master_profile']))
    job_description = payload['job_description']
    selected_tags = payload['selected_tags']
    llm_model = payload['llm_model']
    opti = OptimizeResume(llm_model=llm_model, master_profile=master_profile,
                          job_description=job_description, selected_tags=selected_tags)
    tuned_profile = opti.tune_resume()

    pprint(json.loads(tuned_profile.model_dump_json()))
