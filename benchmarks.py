from llama_index.llms.ollama import Ollama
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import RubricsScore
from ragas.exceptions import RagasOutputParserException
from ragas.dataset_schema import SingleTurnSample

import asyncio


async def evaluate_optimized_resume(
    job_description: str, base_resume: str, optimized_resume: str
):
    """
    Evaluates an optimized resume against a job description and base resume using an LLM-based scoring system.

    Parameters:
    - job_description (str): The job description.
    - base_resume (str): The original, unoptimized resume.
    - optimized_resume (str): The improved resume for evaluation.

    Returns:
    - dict: Contains the rubric score and model-wise feedback.
    """
    print("Loading LLMs")
    models = ["llama3.1", "gemma3:4b"]
    results = {}

    for model in models:
        try:
            evaluator_llm = LlamaIndexLLMWrapper(llm=Ollama(model))
            print(f"Loaded LLM: {model}")

            sample = SingleTurnSample(
                reference=f"""
                job_description:
                -----------------------
                {job_description}
                
                initial_resume:
                -----------------------
                {base_resume}
                """,
                response=f"""
                optimized_resume:
                -----------------------
                {optimized_resume}
                """,
            )

            rubrics = {
                "score1_description": "The resume is **poorly structured, generic, and does not align** with the job description. It lacks key technical skills, measurable impact, and clarity. It may introduce irrelevant or incorrect information, making it **unsuitable for job applications**.",
                "score2_description": "The resume **contains relevant details** but lacks strong emphasis on **key skills, quantifiable achievements, or ATS optimization**. The improvements over the original are **moderate**, but it still needs **better structuring and clarity** to be effective.",
                "score3_description": "The resume is **well-structured, job-aligned, ATS-friendly, and highlights key skills and quantifiable achievements**. It **significantly improves upon the original** and is **ready for job applications** without major revisions.",
            }

            scorer = RubricsScore(rubrics=rubrics, llm=evaluator_llm)
            rubric_score = await scorer.single_turn_ascore(sample)

            results[model] = {"rubric_score": rubric_score}

        except RagasOutputParserException:
            print(f"Model {model} failed")
            results[model] = {"error": "Ragas output parsing error"}

        except Exception as e:
            print(f"Other error {e} for model {model}")
            results[model] = {"error": str(e)}
    
    all_scores = [i['rubric_score'] for i in results.values()]
    results['Average'] =   sum(all_scores) / len(all_scores)

    return results


if __name__ == "__main__":
    # Define the job description and initial resume (same for all calls)
    job_description = """
    Software Engineer – Machine Learning (ML)  
    Responsibilities:  
    - Develop and optimize deep learning models for real-world applications  
    - Collaborate with cross-functional teams to deploy ML solutions  
    - Improve model efficiency and scalability using PyTorch/TensorFlow  
    - Work with large-scale datasets and cloud infrastructure (AWS/GCP)  
    - Strong proficiency in Python, data structures, and algorithms  
    - Experience with CI/CD and MLOps best practices  
    """

    initial_resume = """
    John Doe  
    - Worked on machine learning models in Python.  
    - Familiar with deep learning concepts.  
    - Experience with cloud computing and software engineering.  
    - Knowledge of version control systems.  
    """

    # The different optimized resumes for different rubric scores
    optimized_resumes = [
    {
        "resume": """ 
        John Doe  
        - Worked on Python Snakes  
        - Knows cloudy with a chance of meatballs.  
        - Interested in machine learning.  
        - Familiar with coding.  
        """,
        "expected_score": 1
    },
    {
        "resume": """ 
        John Doe  
        - Worked on Python-based ML models and trained neural networks.  
        - Used AWS for cloud computing and model deployment.  
        - Familiar with TensorFlow and PyTorch.  
        - Understands version control and basic CI/CD principles.  
        """,
        "expected_score": 2
    },
    {
        "resume": """ 
        John Doe  
        - Built machine learning models using TensorFlow and PyTorch, achieving a moderate increase in efficiency.  
        - Worked with AWS cloud services to deploy models and manage large-scale datasets.  
        - Implemented CI/CD pipelines for automated model deployment and monitoring.  
        - Applied data structures and algorithms to optimize model performance.  
        """,
        "expected_score": 2
    },
    {
        "resume": """ 
        John Doe  
        - Developed and optimized deep learning models using PyTorch and TensorFlow, improving model accuracy by 10% and reducing training time by 20%.  
        - Deployed scalable ML solutions on AWS, reducing cloud costs by 15% and improving inference speed.  
        - Built data pipelines for large-scale datasets, enhancing data processing efficiency by 30%.  
        - Implemented CI/CD for automated ML model deployment and monitoring, reducing downtime.  
        """,
        "expected_score": 3
    },
    {
        "resume": """ 
        John Doe  
        - Designed and optimized deep learning models using PyTorch and TensorFlow, achieving a 25% reduction in training time and a 15% improvement in accuracy.  
        - Led the development and deployment of scalable ML solutions on AWS/GCP, reducing cloud computing costs by 20% and improving inference speed by 40%.  
        - Built robust data pipelines, processing terabytes of data efficiently with distributed computing.  
        - Spearheaded MLOps automation, implementing CI/CD workflows and automated monitoring, reducing deployment time by 50%.  
        - Collaborated with cross-functional teams to integrate ML models into production systems, ensuring seamless deployment.  
        """,
        "expected_score": 3
    }
]

    # Call the function for each optimized resume
    async def test_rubric_scoring():
        for i, entry in enumerate(optimized_resumes):
            print(f"\nEvaluating Optimized Resume {i + 1} (Expected Score: {entry['expected_score']})\n")
            score = await evaluate_optimized_resume(
                job_description, initial_resume, entry["resume"]
            )
            print(f"Rubric Score: {score}\n")

    # Run the test
    asyncio.run(test_rubric_scoring())
