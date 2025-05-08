import time
import yaml
import pandas as pd
from evaluation.evaluator_base import ResumeEvaluationEngine
from dataclasses import dataclass
from typing import Literal, Optional
import jinja2
from llm_provider import LLMClient


DEBUG = False


@dataclass
class ModelExperimentConfiguration:
    backend: Literal["ollama", "gemini"]
    model_name: Optional[str]
    prompts_path: str


class ExperimentHarness(ResumeEvaluationEngine):
    def __init__(self, csv_path, *args, **kwargs):

        self.df = pd.read_csv(csv_path)
        super().__init__(*args, **kwargs)

    def run_experiment(self, experiment: ModelExperimentConfiguration):

        df_copy = self.df.copy(deep=True)
        gen_func = LLMClient(
            backend=experiment.backend, model=experiment.model_name
        )
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

        return df_copy

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
        backend="ollama", model_name="smollm2", prompts_path=prompts_path
    )
    harness = ExperimentHarness(csv_path="./evaluator_harness.csv")
    output = harness.run_experiment(experiment)
    output.to_csv("./experiment_output.csv")
