# from pprint import pprint
import os
from dotenv import load_dotenv
load_dotenv()
from evaluation.evaluator_backend import GeminiBackend


metrics_schema = {
    "type": "object",
    "properties": {
        "Factual Accuracy": {
            "type": "string",
            "enum": [
                "Factual",
                "Deceptive"
            ],
            "description": "One of: Factual, Deceptive"
        },
        "Alignment": {
            "type": "string",
            "enum": [
                "Misaligned",
                "Neutral",
                "Well Aligned"
            ],
            "description": "One of: Misaligned, Neutral, Well Aligned"
        },
        "Section Length": {
            "type": "string",
            "enum": [
                "Too Short",
                "Optimal",
                "Too Long"
            ],
            "description": "One of: Too Short, Optimal, Too Long"
        },
        "Grammar": {
            "type": "string",
            "enum": [
                "Needs Improvement",
                "Acceptable",
                "Polished"
            ],
            "description": "One of: Needs Improvement, Acceptable, Polished"
        }
    },
    "required": [
        "Factual Accuracy",
        "Alignment",
        "Section Length",
        "Grammar"
    ],
}


user_prompt = """Please evaluate the improved resume section based on the four metrics: Factual Accuracy, Alignment, Section Length, and Grammar.
Use the following inputs:

Original Section (reference for truthfulness and factuality):
Developed a RESTful API in Flask for a food delivery app, integrating JWT authentication and PostgreSQL database.

Job Description (reference for alignment):
Looking for a backend engineer skilled in Python, REST API development, and relational databases.

Improved Section (the content to evaluate):
Built secure, scalable REST APIs using Flask, implemented JWT-based authentication, and integrated PostgreSQL for data management.

For each of the four metrics, assign a score from the available options and briefly justify your choice. Use the definitions provided in the system prompt as your guide.
"""

system_prompt = """
You are an expert resume evaluation assistant. Your role is to assess improved resume sections using four key metrics, comparing them against the original section and the job description. Each metric has well-defined categories and descriptions to guide your evaluation.
Use the original content as the source of truth for assessing factual accuracy. For each metric, choose the most appropriate score based on the content quality and relevance. Justify your choices with a short explanation.

Evaluation metrics:
Metric: factual_accuracy
Definition: Assesses the truthfulness and reliability of the improved resume content. It ensures that all claims are grounded in the original input, accurately reflecting the candidate's real skills, experiences, and achievements. The input content is considered the sole source of truth.
Options:
  1. Factual: The resume is accurate and credible, with truthful, well-supported claims that align with the candidate's actual skills and experiences.
  2. Deceptive: The resume contains inaccuracies, exaggerations, or vague claims that lack clarity or verifiable evidence, potentially misleading employers.

Metric: alignment
Definition: Measures how effectively the resume section aligns with the job description. A well-aligned section highlights job-relevant skills and experiences while minimizing or excluding unrelated or less important details.
Options:
  1. Misaligned: The resume section fails to align with the job description. It emphasizes irrelevant or less important content while underrepresenting key job-specific skills and qualifications, leading to a poor fit for the role.
  2. Neutral: The resume section includes some relevant information but lacks a focused alignment with the job description. It does not emphasize important qualifications effectively, yet avoids major misrepresentation or irrelevance.
  3. Well Aligned: The resume section strongly aligns with the job description, emphasizing key qualifications, skills, and experiences that are most relevant to the role while minimizing less important or unrelated content.

Metric: section_length
Definition: Evaluates whether the resume section is appropriately concise and well-balanced. It should present all essential information without being overly brief or unnecessarily verbose, supporting readability and impact.
Options:
  1. Too Short: The resume section is too concise, omitting important context or details. Key information may be lost, weakening the overall clarity or impact.
  2. Optimal: The resume section is clear, concise, and well-balanced. It includes all essential information without redundancy, making it easy to skim and impactful.
  3. Too Long: The resume section is overly verbose or cluttered, including unnecessary details that reduce clarity and make it harder to identify the most important points.

Metric: grammar
Definition: Assesses the correctness and professionalism of the resume's grammar, spelling, and formatting. Strong writing mechanics enhance clarity, readability, and the candidate's credibility.
Options:
  1. Needs Improvement: The resume contains noticeable grammar, spelling, or formatting errors that impact readability and professionalism.
  2. Acceptable: The resume has minor grammatical or spelling issues, but they don't significantly affect readability or overall presentation.
  3. Polished: The resume is well-written with excellent grammar, spelling, and formatting, presenting the content clearly and professionally.

"""

gemini = GeminiBackend()
response= gemini.evaluate(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    json_schema=metrics_schema
)

print(response)


# prompt = """List a few popular cookie recipes in JSON format.

# Use this JSON schema:

# Recipe = {'recipe_name': str, 'ingredients': list[str]}
# Return: list[Recipe]"""

# key = os.getenv("GEMINI_API_KEY")

# print(key)

# # client = genai.Client(api_key="AIzaSyCvVuvZoWDOhd_SN-12mRs8hi0WkRB4rJI")
# # response = client.models.generate_content(
# #     model='gemini-2.0-flash',
# #     contents=prompt,
# # )

# # # Use the response as a JSON string.
# # print(response.text)