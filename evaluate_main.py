from evaluation.evaluator_harness import ExperimentHarness, ModelExperimentConfiguration
from dotenv import load_dotenv
load_dotenv()

# from evaluation.evaluator_tester import EvaluatorTester
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

# csv_path = './evaluation/evaluator_tester.csv'
# csv_path = './data/rizzume_optimization_v1.csv'
# ev_tester = EvaluatorTester(eval_backend='gemini'
#                             # ,model='smollm2'
#                             )
# # ev_df = ev_tester.evaluate_from_csv(csv_path)
# # ic(ev_df)
# analysis_results = "C:/Users/viren/Desktop/Rizzume/Code/rizzume_ml/eval_test/eval_05_04_2025_15_47_07_gemini_gemini-2.0-flash.csv"
# ev_tester.analyze_results(analysis_results)

prompts_path = "./inference/prompts.yaml"
benchmark_path = './data/rizzume_optimization_v1_benchmark.csv'
experiments = [
    ModelExperimentConfiguration(
        
        experiment_name='gemini',backend="gemini", model_name="gemini-2.0-flash", prompts_path=prompts_path
    ),
    ModelExperimentConfiguration(
        experiment_name= "smollm2",
        backend="ollama", model_name="smollm2", prompts_path=prompts_path
    ),
    ModelExperimentConfiguration(
        experiment_name="llama3.2",
        backend="ollama", model_name="llama3.2", prompts_path=prompts_path
    )]
experiment_dir = "C:/Users/viren/Desktop/Rizzume/Code/rizzume_ml/experiment_results/"
harness = ExperimentHarness(experiment_prefix='first',csv_path=benchmark_path,all_experiment_results_dir=experiment_dir)
output = harness.run_experiments(experiments,save_individual_experiments=True)
output.to_csv('./experiment_output.csv')
