import json
from classify import predict_item
from    profile_model import MasterProfile

def template_projects(item):

    formatted = f"""
    "Title: {item['title']}
    Organization: {item['organization']}
    Description: {item['description']}
    """

    return formatted



def template_experiences(item):


    formatted = f"""
        "Role: {item['role']}
        Company: {item['company']}
        Company Summary: {item['about the company']}
        Onsite / Remote : {item['onsite / remote']}
        Description: {item['description']}
        """

    return formatted


def template_certifications(item):
    formatted = f"""
    "Title: {item['title']}
    Organization: {item['organization']}
    """

    return formatted

def template_achievements(item):
    formatted = f"""
    "Award Title: {item['Award Title']}
    Description: {item['Description']}
    """

    return formatted


def predict_on_master_profile(master_profile:MasterProfile,all_tags):

    master_profile = json.loads(master_profile.model_dump_json())
    allowed_sections = ['projects', 'experience',
                        'certifications', 'achievements']
    processing_templates = {"projects": template_projects, "experience": template_experiences,
                            "certifications": template_certifications, "achievements": template_achievements}
    for section in allowed_sections:
        for i in range(len(master_profile[section])):
            formatted_item = processing_templates[section](master_profile[section][i])
            tags = predict_item(formatted_item,all_tags)
            print('-------------------')     
            print("Input Item")
            print(formatted_item)
            print()
            print(tags)
            

            print("DEBUG")
            print(master_profile[section][i])
            print(master_profile[section][i]['tags'],
                  type(master_profile[section][i]['tags']))
            master_profile[section][i]['tags'].extend(tags)
    
    return master_profile
    
       