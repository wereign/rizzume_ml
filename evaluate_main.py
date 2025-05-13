import os
from evaluation.evaluator_harness import ExperimentHarness, ModelExperimentConfiguration
from dotenv import load_dotenv

load_dotenv()

all_prompts_path = ["./prompts/single_shot_prompts.yaml","./prompts/zero_shot_prompts.yaml","./prompts/few_shot_prompts.yaml",]
benchmark_path = "./data/rizzume_optimization_v2_80.csv"
all_experiments = []

for prompt_path in all_prompts_path:
    experiments = [
        ModelExperimentConfiguration(
            experiment_name="gemini",
            backend="gemini",
            model_name="gemini-2.0-flash",
            prompts_path=prompt_path,
        ),
        ModelExperimentConfiguration(
            experiment_name="smollm2",
            backend="ollama",
            model_name="smollm2",
            prompts_path=prompt_path,
        ),
        ModelExperimentConfiguration(
            experiment_name="llama3.2",
            backend="ollama",
            model_name="llama3.2",
            prompts_path=prompt_path,
        ),
    ]
    all_experiments.extend(experiments)
experiment_dir = "C:/Users/viren/Desktop/Rizzume/Code/rizzume_ml/experiment_results/"
harness = ExperimentHarness(
    experiment_prefix="prompt_model_variations2",
    csv_path=benchmark_path,
    all_experiment_results_dir=experiment_dir,
)
output = harness.run_experiments(all_experiments, save_individual_experiments=True)
output.to_csv("./experiment_output.csv")
