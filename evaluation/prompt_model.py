from enum import Enum
import json
from pprint import pprint
from typing import Any, Dict
from jinja2 import Template
import ollama

DEBUG = False
BASE_URL = "http://localhost:11434"

def construct_metrics_string(metrics_json):
    """
    Constructs a string representation of the metrics from a JSON object.
    
    Args:
        metrics_json (dict): A dictionary containing the metrics.
        
    Returns:
        str: A formatted string representation of the metrics.
    """
    metrics_string = ""
    for metric, details in metrics_json.items():
        metrics_string += f"Metric: {metric}\n"
        metrics_string += f"Definition: {details['description']}\n"
        metrics_string += "Options:\n"
        for idx, (key, description) in enumerate(details['options'].items(), start=1):
            metrics_string += f"  {idx}. {key}: {description}\n"
        metrics_string += "\n"
    return metrics_string.strip()
        
def load_metrics_from_json(path):
    """
    Loads metrics from a JSON file.
    
    Args:
        path (str): The path to the JSON file.
        
    Returns:
        dict: A dictionary containing the metrics.
    """
    try:
        with open(path, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def load_prompts(yaml_path):
    """
    Loads prompts from a YAML file.
    
    Args:
        yaml_path (str): The path to the YAML file.
        
    Returns:
        dict: A dictionary containing the prompts.
    """
    import yaml
    try:
        with open(yaml_path, "r") as f:
            prompts = yaml.safe_load(f)
        return prompts
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def construct_final_prompt(input_experience:str, job_description:str, output_experience:str, metrics:dict, prompts:dict):
    """
    Uses all the available information to construct the final prompt for the model.

    Args:
        input_experience (str): The experience details of the input.
        job_description (str): The job description to match against.
        output_experience (str): The expected output experience.
        metrics (dict): A dictionary containing relevant metrics, along with their descriptions and options.
    """

    evaluation_metrics_string = construct_metrics_string(metrics)

    system_prompt_template = Template(prompts.get("system_prompt", "System prompt not found."))
    user_prompt_template = Template(prompts.get("user_prompt", "User prompt not found."))

    system_prompt = system_prompt_template.render(evaluation_metrics_string=evaluation_metrics_string)
    user_prompt = user_prompt_template.render(input_experience=input_experience, job_description=job_description, output_experience=output_experience)

    print("==="*20)
    print("System Prompt")
    print(system_prompt)
    print()
    print("==="*20)
    print("User Prompt")
    print(user_prompt)
    return system_prompt, user_prompt



def create_enum_from_metric_dict(metric_name: str, metric_options: dict, debug=DEBUG):
    metric_strings = []

    # constructing an enum_dict
    enum = Enum(metric_name, tuple(metric_options.items()), type=str)

    for key, description in metric_options.items():
        metric_strings.append(f"{key}: {description}")

    if debug:
        print("-" * 50)
        print(f"Metric : {metric_name}")
        print(f"Score Options : \n {metric_strings}")
        print()

    return enum


def construct_output_options(metrics: dict):
    """
    Constructs output options and a JSON schema including explanations for each metric.

    Args:
        metrics (dict): A dictionary containing the metrics.

    Returns:
        dict: A JSON Schema dictionary with structured output format including justifications.
    """
    output_options = {}
    enum_dict = {}

    for metric, details in metrics.items():
        output_options[metric] = details["options"]
        enum_dict[metric] = create_enum_from_metric_dict(
            metric, details["options"]
        )  # Assuming this is defined elsewhere

    json_schema = {
        "type": "object",
        "properties": {},
        "required": list(metrics.keys()) + ["Justifications"],
        "additionalProperties": False,
    }

    # Add primary score fields
    for category in metrics:
        options = metrics[category]["options"]
        json_schema["properties"][category] = {
            "type": "string",
            "enum": list(options.keys()),
            "enumDescriptions": list(options.values()),
            "description": f"One of: {', '.join(options.keys())}",
        }

    # Add explanation fields inside a nested object
    json_schema["properties"]["Justifications"] = {
        "type": "object",
        "properties": {
            category: {
                "type": "string",
                "description": f"Explanation for the {category} score.",
            }
            for category in metrics
        },
        "required": list(metrics.keys()),
        "additionalProperties": False,
    }

    return json_schema


def evaluate_resume_with_structure(
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    model: str = "gemma3:1b",
) -> Dict[str, Any]:
    """
    Prompts an Ollama model with structured output using a system prompt, user prompt, and JSON schema.

    Args:
        system_prompt (str): The system-level instructions for the model.
        user_prompt (str): The user-level prompt including inputs.
        json_schema (Dict): The JSON Schema to enforce structure.
        model (str): The model to use (default is "gemma3:1b").

    Returns:
        Dict[str, Any]: The parsed structured response from the model.
    """
    try:
        client = ollama.Client(host=BASE_URL)
        response = client.chat(
            model= model,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            ],
            format=json_schema,
            options={"temperature": 0.2},
        )
        print(response.message.content)
        return json.loads(response.message.content)

    except Exception as e:
        print("Error communicating with Ollama or parsing response:", e)
        return {}

def eval_model(input_experience, job_description, output_experience):
    metrics = load_metrics_from_json("./metrics.json")
    prompts = load_prompts("./prompts.yaml")
    system_prompt, user_prompt = construct_final_prompt(input_experience, job_description, output_experience, metrics, prompts)
    output_json_schema = construct_output_options(metrics)
    response = evaluate_resume_with_structure(system_prompt, user_prompt, output_json_schema)
    pprint(response)

    return response

    


if __name__ == "__main__":
    input_experience = "Software Engineer with 5 years of experience in Python and Java."
    job_description = "Looking for a Software Engineer with experience in Python, Java, and cloud technologies."
    output_experience = "Software Engineer with 5 years of experience in Python, Java, and AWS."
    eval_model(input_experience, job_description, output_experience)
    
