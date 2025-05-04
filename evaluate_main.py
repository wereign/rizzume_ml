from dotenv import load_dotenv
from icecream import ic
load_dotenv()
from evaluation.evaluator_tester import EvaluatorTester

# input_exp = "Software Engineer with 5 years of experience in Python, Java and AWS."
# job_desc = "Looking for a Software Engineer with experience in Python, Java, and cloud technologies."
# output_exp = (
# "Software Engineer of 5 years into experience onto Python, Java, and AWS."
# )
# ree = ResumeEvaluationEngine(metrics_path='./evaluation/metrics.json',
#                        prompt_path='./evaluation/prompts.yaml',
#                        eval_backend='ollama',model='llama3.2')

# result = ree.evaluate(input_experience=input_exp,
#                       job_description=job_desc,
#                       output_experience=output_exp)


# ic(input_exp)
# ic(job_desc)
# ic(output_exp)
# ic(result)


csv_path = './evaluation/evaluator_tester.csv'
csv_path = './data/rizzume_optimization_v1.csv'
ev_tester = EvaluatorTester(eval_backend='gemini'
                            # ,model='smollm2'
                            )
ev_df = ev_tester.evaluate_from_csv(csv_path)

ic(ev_df)
ev_tester.analyze_results()