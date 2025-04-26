from google import genai
from pydantic import BaseModel


client = genai.Client(api_key="AIzaSyCvVuvZoWDOhd_SN-12mRs8hi0WkRB4rJI")

system_prompt = """
    You are a resume optimization assistant.Your task is to refine descriptions of academic or personal projects in a resume to align them with a specific job description.
    1. Ensure the rewritten content is:
      - Gramatically correct, and professional in tone.
      - Factually accurate (do not add details not found in the original).
      - Concise, especially if the original description is brief.
      - Focused on relevant technical skills, tools, impact, and achievements that match the job description.
      - structured with points in descending order of relevance to the job description from the top to the bottom
    2. After rewriting, assign a relevance score from 1 to 10 based on how well the project aligns with the job role.
    3. Provide a gap analysis report on the key aspects of the job requirement missing from this experience
    4. Provide a structured explanation for the improvements you have made
    
"""

job_description = """
Collaborate with cross-functional teams on project delivery.
Develop machine learning models using Python & TensorFlow.
Optimize data pipelines for efficiency & accuracy.
Strong understanding of machine learning and deep learning principles and algorithms.
Experience in developing and implementing generative Al models and algorithms.
Proficiency in programming languages such as Python, TensorFlow, and PyTorch.
Ability to work with large datasets and knowledge of data preprocessing techniques.
Familiarity with natural language processing (NLP) and computer vision for generative Al applications.
Experience in building and deploying generative Al systems in real-world applications.
Utilize advanced machine learning techniques to develop and train generative Al models.
Decode the Requirements to functional modules and product features
Pipeline building integration and testing following the SDLC process
"""


experience = {
    "role": "Data Scientist Intern",
            "company": "Grubhub",
            "location": "Pune, India",
            "start_date": "2021-06-01T00:00:00.000+00:00",
            "end_date": "2021-12-01T00:00:00.000+00:00",
            "mode": "Remote",
            "description": "- Designed and implemented over 40 machine-learning models for different programs and projects"
            "- Verified results of algorithms to predict future occurrences using real-world programs data with 82% precision"
            "- Extracted raw data from Twitter APIs and analyzed tweets to generate analysis showing trends in public opinion regarding policy changes"
            "- Developed a Java application that performed pattern analysis of criminal incidents to help identify and visualize hotspots (vulnerable areas) in the city"
            "- Developed a TensorFlow application to detect the sentiment of the pubic opinion on the policy changes. The deep learning dank model performed with a dope accuracy of 90%",
}

exp_string = "\n".join([f"{key}:{experience[key]}" for key in experience])

user_query = f"""
Job Description
----
{job_description}

Experience 
---
{exp_string}

"""


class GeminiResponse(BaseModel):
    modified_response : str
    relevance_score : int
    gap_analysis: str
    explanation: str




response = client.models.generate_content(
    model="gemini-2.0-flash", contents=user_query,
    config=genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.5,
        response_mime_type='application/json',
        response_schema= GeminiResponse
    )
)

print(response.text)

# Use the response as a JSON string.
print(response.text)
