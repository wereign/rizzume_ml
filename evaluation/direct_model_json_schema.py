import json
from typing import Dict


def generate_json_schema(metric_dict: Dict[str, Dict[str, str]]) -> Dict:
    schema = {
        "type": "object",
        "properties": {},
        "required": list(metric_dict.keys()),
        "additionalProperties": False,
    }

    for category, options in metric_dict.items():
        schema["properties"][category] = {
            "type": "string",
            "enum": list(options.keys()),
            "description": f"One of: {', '.join(options.keys())}",
        }

    return schema


# Sample input dictionary (can be replaced with any similar structure)
input_dict = {
    "Factual Accuracy": {
        "Factual": "The resume is accurate and credible, with truthful, well-supported claims that align with the candidate's actual skills and experiences.",
        "Deceptive": "The resume contains inaccuracies, exaggerations, or vague claims that lack clarity or verifiable evidence, potentially misleading employers.",
    },
    "Alignment": {
        "Misaligned": "The resume section fails to align with the job description. It emphasizes irrelevant or less important content while underrepresenting key job-specific skills and qualifications, leading to a poor fit for the role.",
        "Neutral": "The resume section includes some relevant information but lacks a focused alignment with the job description. It does not emphasize important qualifications effectively, yet avoids major misrepresentation or irrelevance.",
        "Well Aligned": "The resume section strongly aligns with the job description, emphasizing key qualifications, skills, and experiences that are most relevant to the role while minimizing less important or unrelated content.",
    },
    "Section Length": {
        "Too Short": "The resume section is too concise, omitting important context or details. Key information may be lost, weakening the overall clarity or impact.",
        "Optimal": "The resume section is clear, concise, and well-balanced. It includes all essential information without redundancy, making it easy to skim and impactful.",
        "Too Long": "The resume section is overly verbose or cluttered, including unnecessary details that reduce clarity and make it harder to identify the most important points.",
    },
    "Grammar": {
        "Needs Improvement": "The resume contains noticeable grammar, spelling, or formatting errors that impact readability and professionalism.",
        "Acceptable": "The resume has minor grammatical or spelling issues, but they don't significantly affect readability or overall presentation.",
        "Polished": "The resume is well-written with excellent grammar, spelling, and formatting, presenting the content clearly and professionally.",
    },
}

# Generate the schema
json_schema = generate_json_schema(input_dict)

# Print the JSON Schema
print(json.dumps(json_schema, indent=4))

with open('./resume_metrics.json','w') as f:
    json.dump(json_schema,f)
