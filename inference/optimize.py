from pprint import pprint
from typing import List
import yaml
import json
from copy import deepcopy
from profile_model import MasterProfile
from llm_provider.llm import LLMClient


ALL_OLLAMA_MODELS = ["llama3.2", "smollm2", "gemma3:4b", "gemma3:1b", "command-r7b"]
ALL_GEMINI_MODELS = ['gemini-2.0-flash']
class OptimizeResume():
    def __init__(self, llm_model, master_profile: MasterProfile, job_description: str, selected_tags: list, prompts_path: str = './prompts.yaml'):

        self.model_name = llm_model
        if self.model_name in ALL_OLLAMA_MODELS:
            backend = 'ollama'
        elif self.model_name in ALL_GEMINI_MODELS:
            backend = 'gemini'

        self.client = LLMClient(backend=backend,model=self.model_name)

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

        filtered_profile:MasterProfile = deepcopy(self.master_profile)
        master_profile:MasterProfile = self.master_profile
        selected_tags = self.selected_tags

        filtered_profile.skills = master_profile.skills
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

    def optimize_item(self,job_description, item):
        print("Optimizing Item")
        prompts = self.prompts["optimize_item"]
        system_prompt = prompts["system_prompt"]
        user_prompt = prompts['user_prompt'].format(jd=job_description,item=item)
        
        print(f"Sent item to {self.model_name}")
        response = self.client.generate(system_prompt,user_prompt)
        optimized_content = response.message.content
        
        print(f"Response from {self.model_name}")
        return optimized_content

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
