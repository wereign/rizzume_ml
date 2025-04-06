from transformers import pipeline
import json
from profile_model import MasterProfile, Experience, Certification, Achievement, Publication

PREDICTION_THRESHOLD = 0.45
classifier = pipeline(model="facebook/bart-large-mnli", device='cuda:0')

def template_projects(item):

    formatted = f"""
    "Title: {item['title']}
    Organization: {item['organization']}
    Description: {item['description']}
    """

    return formatted


def template_experiences(item:Experience):


    formatted = f"""
        "Role: {item['role']}
        Company: {item['company']}
        Onsite / Remote : {item['mode']}
        Description: {item['description']}
        """

    return formatted


def template_certifications(item:Certification):
    formatted = f"""
    "Title: {item['title']}
    Organization: {item['organization']}
    """

    return formatted

def template_achievements(item:Achievement):
    formatted = f"""
    "Award Title: {item['award_title']}
    Description: {item['description']}
    """

    return formatted

def template_publications(item:Publication):
    formatted = f"""
    "Award Title: {item['title']}
    Description: {item['publisher']}
    """

    return formatted


def predict_item(item, user_tags, classifier=classifier, log=False, top_n=2, selection='threshold'):
    """
    Returning a list of suggested tags from the tags the user has created. 
    Threshold filtering to select the tags right now.

    Args:
        block (str): str representation of a block in the Resume 
        user_tags (list): list of all tags created by the user
        classifier (pipeline, optional): Classification Model. Defaults to classifier.
        log (bool, optional): Prints the prediction results. Setting up Logging in the future. Defaults to True.
    """
    # ISSUE: if the number of relevant tags is less than the top_n value, then some predictions will be bogus.
    # ISSUE: threshold value needs to be experiemnted with
    # SOLUTION: Find the point where the confidence drops off drastically, and then use that as the threshold.

    assert selection in ['top_n', 'threshold']
    if len(user_tags) > 0:
        results = classifier(item, candidate_labels=user_tags)
        preds = list(zip(results['labels'], results['scores']))

        if log:
            print("".join([f"{x[0]}: {x[1]}\n" for x in zip(
                results['labels'], results['scores'])]))

        if selection == 'top_n':
            preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

        elif selection == "threshold":
            preds = [x[0]
                     for x in filter(lambda x: x[1] > PREDICTION_THRESHOLD, preds)]

        return preds

    else:
        return None


def predict_on_master_profile(master_profile:MasterProfile,all_tags):
    master_profile = json.loads(master_profile.model_dump_json())
    processing_templates = {"projects": template_projects, "experience": template_experiences,
                            "certifications": template_certifications, "achievements": template_achievements,
                            'publications':template_publications}
    
    for section in processing_templates:
        for i in range(len(master_profile[section])):
            formatted_item = processing_templates[section](master_profile[section][i])
            tags = predict_item(formatted_item,all_tags)
            master_profile[section][i]['tags'] = tags
            print('-------------------')     
            print("Input Item")
            print(formatted_item)
            print()
            print(tags)

            print("DEBUG")
            print(master_profile[section][i])
            print(master_profile[section][i]['tags'])
            master_profile[section][i]['tags'].extend(tags)
    
    return master_profile
    
