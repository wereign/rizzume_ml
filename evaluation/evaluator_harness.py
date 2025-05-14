from datetime import datetime
import os
from pathlib import Path
from pprint import pprint
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
    def __init__(self, experiment_prefix, csv_path, all_experiment_results_dir, *args, **kwargs):
        self.experiment_prefix = experiment_prefix

        self.experiment_results_dir = f"{all_experiment_results_dir}/{experiment_prefix}"
        os.makedirs(f"{all_experiment_results_dir}/{experiment_prefix}",exist_ok=True)
        self.checkpoint_dir = f"{self.experiment_results_dir}/{experiment_prefix}_checkpoints"
        self.output_dir = f"{self.experiment_results_dir}/{experiment_prefix}_outputs"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.df = pd.read_csv(csv_path)
        super().__init__(*args, **kwargs)


    def optimize_experiments(
        self,
        experiments: Union[
            ModelExperimentConfiguration, List[ModelExperimentConfiguration]
        ],
    ) -> pd.DataFrame:
        if isinstance(experiments, ModelExperimentConfiguration):
            experiments = [experiments]

        all_optimized_dfs = []

        for experiment in experiments:
            print("\n[OPTIMIZATION] Running:", experiment)

            df_copy = self.df.copy(deep=True)
            file_suffix = self.get_file_suffix(experiment)
            output_path = os.path.join(self.output_dir, f"{file_suffix}.csv")

            optimized_df = self.optimize_df(
                df=df_copy,
                prompts_path=experiment.prompts_path,
                backend=experiment.backend,
                model_name=experiment.model_name,
                checkpoint_suffix=file_suffix,
            )

            optimized_df.to_csv(output_path, index=False)
            all_optimized_dfs.append(optimized_df)

        # Save combined DataFrame
        combined_df = pd.concat(all_optimized_dfs, ignore_index=True)
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        combined_output_path = os.path.join(
            self.experiment_results_dir,
            f"{self.experiment_prefix}_combined_optimized_{timestamp}.csv",
        )
        combined_df.to_csv(combined_output_path, index=False)
        print(f"\n[COMBINED OUTPUT] Saved to: {combined_output_path}")

        return combined_df

    def evaluate_experiments(
        self,
        experiments: Union[ModelExperimentConfiguration, List[ModelExperimentConfiguration]],
        save_individual_experiments: bool = True,
        save_final_experiments: bool = True
    ) -> pd.DataFrame:
        """
        Load existing optimized outputs for experiments and evaluate them.

        Args:
            experiments: List of experiment configurations to evaluate.
            save_individual_experiments: Whether to save evaluated CSVs to checkpoint directory.
            save_final_experiments: Whether to save concatenated result CSV.

        Returns:
            Final concatenated dataframe of evaluated results.
        """
        if isinstance(experiments, ModelExperimentConfiguration):
            experiments = [experiments]

        result_dfs = []

        for experiment in experiments:
            file_suffix = self.get_file_suffix(experiment)
            optimized_csv_path = os.path.join(self.output_dir, f"{file_suffix}.csv")

            if not os.path.exists(optimized_csv_path):
                print(f"[WARNING] Missing optimized output for {file_suffix}, skipping.")
                continue

            print(f"[EVALUATION] Evaluating: {optimized_csv_path}")
            df_to_evaluate = pd.read_csv(optimized_csv_path)

            evaluated_df = self.evaluate_df(
                df=df_to_evaluate,
                checkpoint_suffix=file_suffix,
                save_to_checkpoint=save_individual_experiments
            )
            result_dfs.append(evaluated_df)

        final_df = pd.concat(result_dfs, ignore_index=True)

        if save_final_experiments:
            timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            final_path = os.path.join(self.experiment_results_dir, f"{self.experiment_prefix}_evaluation_{timestamp}.csv")
            final_df.to_csv(final_path, index=False)
            print(f"\n[FINAL EVAL] Results saved to: {final_path}")

        return final_df
    
    def optimize_df(
        self,
        df: pd.DataFrame,
        prompts_path: str,
        backend: Literal["ollama", "gemini"],
        model_name: str,
        checkpoint_suffix: Optional[str] = None,
        save_to_output: bool = True
    ) -> pd.DataFrame:
        """
        Optimizes the input dataframe by generating output_experience using LLM.

        Args:
            df (pd.DataFrame): Unoptimized DataFrame with 'input_experience' and 'job_description'.
            prompts_path (str): Path to YAML file with prompts.
            backend (str): Backend for LLMClient ("ollama" or "gemini").
            model_name (str): Name of the model.
            checkpoint_suffix (str, optional): Optional suffix for filename.
            save_to_output (bool): Whether to save the optimized dataframe to output_dir.

        Returns:
            pd.DataFrame: DataFrame with 'output_experience' and 'duration_seconds' added.
        """
        # prompt_technique = Path(prompts_path).stem
        # time_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        gen_func = LLMClient(backend=backend, model=model_name)
        prompts = self.load_prompts(prompts_path)
        optimize_prompts = prompts["optimize_item"]

        df = df.copy()
        df["model"] = model_name
        df["backend"] = backend
        df["prompt_path"] = prompts_path

        df[["output_experience", "duration_seconds"]] = df.apply(
            lambda x: self.optimize_item(
                x["input_experience"],
                x["job_description"],
                optimize_prompts,
                gen_func,
                debug=DEBUG,
                row_number=x.name,
            ),
            axis=1,
        )

        # if save_to_output:
        #     suffix = f"_{checkpoint_suffix}" if checkpoint_suffix else ""
        #     filename = f"{suffix}_{time_string}.csv"
        #     output_path = os.path.join(self.output_dir, filename)
        #     df.to_csv(output_path, index=False)
        #     print(f"Saved optimized dataframe to {output_path}")

        return df

    def evaluate_df(
        self,   
        df: pd.DataFrame,
        checkpoint_suffix: Optional[str] = None,
        save_to_checkpoint: bool = True
    ) -> pd.DataFrame:
        """
        Evaluates an already optimized dataframe.

        Args:
            df (pd.DataFrame): DataFrame with 'input_experience', 'job_description', and 'output_experience'.
            checkpoint_suffix (str, optional): Optional suffix for checkpoint filename.
            save_to_checkpoint (bool): Whether to save the evaluated dataframe to checkpoint_dir.

        Returns:
            pd.DataFrame: DataFrame with evaluation columns added.
        """
        evaluated_df = self.evaluate_optimization(df.copy(deep=True))

        if save_to_checkpoint:
            time_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            suffix = f"_{checkpoint_suffix}" if checkpoint_suffix else ""
            checkpoint_name = f"evaluate_only_{suffix}_{time_string}.csv"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            evaluated_df.to_csv(checkpoint_path, index=False)
            print(f"Saved evaluated dataframe to {checkpoint_path}")

        return evaluated_df

    def optimize_item(
        self,
        item: str,
        job_description: str,
        optimize_prompts: dict,
        gen_func: LLMClient,
        row_number : int,
        debug: bool = True,
    ) -> str:
        row_number = row_number+1
        print(f"Optimizing Sample {row_number}")
        start_time = time.time()
        if debug:
            result = f"[DEBUG] Optimized: {item} for {job_description}"
        else:
            system_prompt, user_prompt = self.render_prompts(
                optimize_prompts, item, job_description
            )

            for i in range(4):
                try:
                    result = gen_func.generate(
                    system_prompt=system_prompt, user_prompt=user_prompt
                    )
                    break
                except Exception as e:
                    print(f"Error in Optimizing due to {e}")
                    time.sleep((i+1)*25)

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
            print(f"Testing Sample {i}")
            success = False

            if i % 15 == 0:
                print("Compulsory 45 second timeout in evaluation")
                time.sleep(45)

            for j in range(4):
                try:
                    print(f"Test Try: {i}")
                    result = self.evaluate(
                    input_experience = row["input_experience"],
                    job_description=row["job_description"],
                    output_experience=row["output_experience"],
                    )
                    success = True
                    break

                except Exception as e:
                    print(f"Exception in Evaluating due to {e}")
                    time.sleep((j+1)*25)

            if success:
                continue
            else:   
                raise RuntimeError(f"Could not evaluate {i}")            

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

    def get_file_suffix(self, experiment: ModelExperimentConfiguration) -> str:
        """
        Generate a deterministic file suffix for each experiment based on its configuration.
        """
        prompt_technique = Path(experiment.prompts_path).stem
        name_prefix = (
            experiment.experiment_name
            or f"{experiment.backend}_{experiment.model_name}"
        )
        return f"{name_prefix}_{prompt_technique}"


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
