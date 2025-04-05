from pprint import pprint
from llama_index.llms.ollama import Ollama
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import RubricsScore
from ragas.exceptions import RagasOutputParserException
from ragas.dataset_schema import SingleTurnSample
from rubrics import faithfulness_rubric, alignment_rubric,conciseness_rubric,customization_rubric, grammar_rubric
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

            all_scorers = {
                "faithfulness": RubricsScore(rubrics=faithfulness_rubric, llm=evaluator_llm),
                "alignment": RubricsScore(rubrics=alignment_rubric, llm=evaluator_llm),
                "conciseness": RubricsScore(
                    rubrics=conciseness_rubric, llm=evaluator_llm
                ),
                "customization": RubricsScore(
                    rubrics=customization_rubric, llm=evaluator_llm
                ),
                "grammar": RubricsScore(rubrics=grammar_rubric, llm=evaluator_llm),
            }
            all_metrics = list(all_scorers.keys())
            results[model] = {
                metric: await all_scorers[metric].single_turn_ascore(sample)
                for metric in all_metrics
            }

        except RagasOutputParserException:
            print(f"Model {model} failed")
            results[model] = {"error": "Ragas output parsing error"}

        except Exception as e:
            print(f"Other error {e} for model {model}")
            results[model] = {"error": str(e)}

    for metric in all_metrics:
        for model in models:
            print(sum([results[model][metric]for model in models]), len(models))
    
    # Avg Score for each metric across all models
    results["average"] = {
        
        metric: sum([results[model][metric]for model in models]) / len(models)
        for metric in all_metrics
    }



    
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
        - Experienced in technology and software.  
        - Interested in AI and cloud computing.  
        - Familiar with coding and programming concepts.  
        """,
            "expected_scores": {
                "alignment": 1,
                "conciseness": 1,
                "customization": 1,
                "grammar": 2,
            },
        },
        {
            "resume": """ 
        John Doe  
        - Worked on Python-based machine learning projects.  
        - Familiar with AWS and cloud computing principles.  
        - Knows basic deep learning concepts but lacks practical experience.  
        - Understands version control and has used Git for personal projects.  
        """,
            "expected_scores": {
                "alignment": 2,
                "conciseness": 2,
                "customization": 2,
                "grammar": 3,
            },
        },
        {
            "resume": """ 
        John Doe  
        - Built machine learning models using TensorFlow and PyTorch, achieving small efficiency improvements.  
        - Worked with AWS cloud services to deploy models and manage datasets.  
        - Implemented CI/CD pipelines for automated model deployment and monitoring.  
        - Applied data structures and algorithms to optimize model performance.  
        """,
            "expected_scores": {
                "alignment": 3,
                "conciseness": 3,
                "customization": 3,
                "grammar": 4,
            },
        },
        {
            "resume": """ 
        John Doe  
        - Developed and optimized deep learning models using PyTorch and TensorFlow, improving accuracy by 10% and reducing training time by 20%.  
        - Deployed scalable ML solutions on AWS, reducing cloud costs by 15% and improving inference speed.  
        - Built data pipelines for large-scale datasets, enhancing data processing efficiency by 30%.  
        - Implemented CI/CD for automated ML model deployment and monitoring, reducing downtime.  
        """,
            "expected_scores": {
                "alignment": 4,
                "conciseness": 4,
                "customization": 4,
                "grammar": 5,
            },
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
            "expected_scores": {
                "alignment": 5,
                "conciseness": 5,
                "customization": 5,
                "grammar": 5,
            },
        },
    ]

    # Call the function for each optimized resume
    async def test_rubric_scoring():
        for i, entry in enumerate(optimized_resumes[3:]):
            print(
                f"\nEvaluating Optimized Resume {i + 1} (Expected Score: {entry['expected_scores']})\n"
            )
            score = await evaluate_optimized_resume(
                job_description, initial_resume, entry["resume"]
            )
            print("Rubric Score: \n")
            pprint(score)
            print()

    # Run the test
    asyncio.run(test_rubric_scoring())