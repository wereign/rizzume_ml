from datetime import datetime
import os
import time
import yaml
import pandas as pd
from evaluation.evaluator_base import ResumeEvaluationEngine
from dataclasses import dataclass
from typing import List, Literal, Optional, Union
import jinja2
from llm_provider import LLMClient


DEBUG = False


@dataclass
class ModelExperimentConfiguration:
    experiment_name: str
    backend: Literal["ollama", "gemini"]
    model_name: Optional[str]
    prompts_path: str


class ExperimentHarness(ResumeEvaluationEngine):
    def __init__(self, experiment_prefix, csv_path, experiment_results_dir, *args, **kwargs):
        self.experiment_prefix = experiment_prefix
        self.experiment_results_dir = experiment_results_dir
        self.checkpoint_dir = f"{experiment_results_dir}/{experiment_prefix}_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.df = pd.read_csv(csv_path)
        super().__init__(*args, **kwargs)


    def run_experiments(
        self,
        experiments: Union[
            ModelExperimentConfiguration, List[ModelExperimentConfiguration]
        ],
        save_individual_experiments: bool = True,
        save_final_experiments:bool = True
    ):
        """
        Runs one or more experiments. Each experiment generates optimized outputs using a specified model,
        then evaluates them using an LLM-as-a-Judge system.

        Args:
            experiments (Union[ModelExperimentConfiguration, List[ModelExperimentConfiguration]]):
                A single experiment or list of experiments to run.
            save_individual_experiments (bool): Whether to save the results of each experiment as a CSV.

        Returns:
            pd.DataFrame: A concatenated dataframe with inputs, outputs, and evaluation metrics from all experiments.
        """
        if isinstance(experiments, ModelExperimentConfiguration):
            experiments = [experiments]

        result_dfs = []

        for experiment in experiments:
            df_copy = self.df.copy(deep=True)
            gen_func = LLMClient(backend=experiment.backend, model=experiment.model_name)
            prompts = self.load_prompts(experiment.prompts_path)
            optimize_prompts = prompts["optimize_item"]

            df_copy["model"] = experiment.model_name
            df_copy["backend"] = experiment.backend
            df_copy["prompt_path"] = experiment.prompts_path

            df_copy[["output_experience", "duration_seconds"]] = df_copy.apply(
                lambda x: self.optimize_item(
                    x["input_experience"],
                    x["job_description"],
                    optimize_prompts,
                    gen_func,
                    debug=DEBUG,
                ),
                axis=1,
            )

            evaluated_df = self.evaluate_optimization(df_copy)
            result_dfs.append(evaluated_df)

            if save_individual_experiments:
                time_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
                csv_path = f"{self.checkpoint_dir}/{experiment.experiment_name}_{time_string}.csv"
                if not os.path.exists(csv_path):
                    evaluated_df.to_csv(csv_path, index=False)

        final_df = pd.concat(result_dfs, ignore_index=True)

        if save_final_experiments:
                csv_path = f"{self.experiment_results_dir}/{self.experiment_prefix}_{time_string}.csv"
                if not os.path.exists(csv_path):
                    evaluated_df.to_csv(csv_path, index=False)

        return final_df

    def optimize_item(
        self,
        item: str,
        job_description: str,
        optimize_prompts: dict,
        gen_func: LLMClient,
        debug: bool = True,
    ) -> str:
        start_time = time.time()
        if debug:
            result = f"[DEBUG] Optimized: {item} for {job_description}"
        else:
            system_prompt, user_prompt = self.render_prompts(
                optimize_prompts, item, job_description
            )
            result = gen_func.generate(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

        duration = time.time() - start_time  # in seconds

        return pd.Series(
            [result, duration], index=["output_experience", "duration_seconds"]
        )

    def evaluate_optimization(
        self, df:pd.DataFrame):
        output_columns = list(df.columns) +  [f"pred_{i}" for i in self.metric_names]

        for col in output_columns:
            if col not in df.columns:
                df[col] = None

        for i, row in df.iterrows():
            if (i + 1) % 15 == 0:
                print('15 Samples tested. Waiting')
                time.sleep(30)

            result = self.evaluate(
                input_experience = row["input_experience"],
                job_description=row["job_description"],
                output_experience=row["output_experience"],
            )

            for key in result:
                df_key = f"pred_{key}"
                df.at[i, df_key] = str(result[key])

        return df

    @staticmethod
    def render_prompts(optimize_prompts, item, job_description):
        if "system_prompt" in optimize_prompts and "user_prompt" in optimize_prompts:
            system_prompt = optimize_prompts["system_prompt"]
            user_prompt = jinja2.Template(optimize_prompts["user_prompt"]).render(
                jd=job_description, item=item
            )

            return system_prompt, user_prompt

        else:
            print("abe lallu")

    @staticmethod
    def load_prompts(prompts_path):
        with open(prompts_path, "r") as pf:
            prompts = yaml.safe_load(pf)

        return prompts


if __name__ == "__main__":

    prompts_path = (
        r"C:\Users\viren\Desktop\Rizzume\Code\rizzume_ml\inference\prompts.yaml"
    )
    experiment = ModelExperimentConfiguration(
        backend="gemini", model_name="gemini-2.0-flash", prompts_path=prompts_path
    )
    experiment = ModelExperimentConfiguration(
        backend="ollama", model_name="smollm2", prompts_path=prompts_path
    )
    experiment = ModelExperimentConfiguration(
        backend="ollama", model_name="llama3.2", prompts_path=prompts_path
    )
    harness = ExperimentHarness(csv_path="./evaluator_harness.csv")
    output = harness.run_experiment(experiment)
    output.to_csv("./experiment_output.csv")
