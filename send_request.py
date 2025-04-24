import json
from pprint import pprint
import requests

with open('./data.json') as jf:
    master_profile = json.load(jf)

payload = {
    "master_profile": master_profile[0]['master_profile'],
    "llm_model":'gemma3:4b',
    "job_description":"""
        Job title: Data Scientist

        Job description:
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
        """,
    "selected_tags":["Deep Learning",
                     "ML Models",
                    "AI",
                    "Tableau",
                    "Data Science",
                    "Machine Learning", 
                    "Data Analysis"]
}
# API Call with Error Handling
try:
    response = requests.post(
        "http://localhost:8080/optimize_profile", json=payload, timeout=500
    )

    # Check if response status is OK (200)
    if response.status_code == 200:
        try:
            optimized_resume = response.json()
            print("Response:")
            print(type(optimized_resume))
            pprint(optimized_resume)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON response.")
    else:
        print(response.json())
        print(f"API Request failed with status code: {response.status_code}")

except requests.Timeout:
    print("Error: API request timed out.")
except requests.RequestException as e:
    print(f"Error: {e}")
