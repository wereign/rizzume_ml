import os
from evaluation.evaluator_harness import ExperimentHarness, ModelExperimentConfiguration
from dotenv import load_dotenv

load_dotenv()

all_prompts_path = [
    "./prompts/single_shot_prompts.yaml",
    "./prompts/zero_shot_prompts.yaml",
    "./prompts/few_shot_prompts.yaml",
]
benchmark_path = "./data/rizzume_optimization_v2_80.csv"
models = [
    ("llama3.2", "llama3.2"),
    ("smollm2", "smollm2"),
    ("gemma3_1b", "gemma3:1b"),
    ("gemma3_4b", "gemma3:4b"),
    ("gemini", "gemini-2.0-flash"),
]

all_experiments = []
for model in models:
    if model[1] == "gemini-2.0-flash":
        backend = "gemini"
    else:
        backend = "ollama"

    prompt_experiments = []
    for prompts_path in all_prompts_path:
        experiment =ModelExperimentConfiguration(
            experiment_name=model[0],
            backend=backend,
            prompts_path=prompts_path,
            model_name=model[1],
        )
        prompt_experiments.append(experiment)
    
    all_experiments.extend(prompt_experiments)

experiment_dir = "C:/Users/viren/Desktop/Rizzume/Code/rizzume_ml/experiment_results/"
harness = ExperimentHarness(
    experiment_prefix="all_optimization_prompts",
    csv_path=benchmark_path,
    all_experiment_results_dir=experiment_dir,
)
output = harness.optimize_experiments(all_experiments)
eval_output = harness.evaluate_experiments(all_experiments)
