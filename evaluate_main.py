from dotenv import load_dotenv
from evaluation.evaluate import ResumeEvaluationEngine
from icecream import ic
load_dotenv()

input_exp = "Software Engineer with 5 years of experience in Python, Java and AWS."
job_desc = "Looking for a Software Engineer with experience in Python, Java, and cloud technologies."
output_exp = (
"Software Engineer of 5 years into experience onto Python, Java, and AWS."
)
ree = ResumeEvaluationEngine(metrics_path='./evaluation/metrics.json',
                       prompt_path='./evaluation/prompts.yaml',
                       eval_backend='ollama',model='llama3.2')

result = ree.evaluate(input_experience=input_exp,
                      job_description=job_desc,
                      output_experience=output_exp)


ic(input_exp)
ic(job_desc)
ic(output_exp)
ic(result)


