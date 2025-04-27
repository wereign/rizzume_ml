import json
from jinja2 import Template
import yaml
from evaluation.evaluator_backend import ResumeEvaluator


class MetricProcessor:
    def __init__(self, metrics_path: str):
        self.metrics_path = metrics_path
        self.metrics = self.load_metrics()

    def load_metrics(self) -> dict:
        try:
            with open(self.metrics_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics JSON file: {e}")
            return {}

    def construct_metrics_string(self) -> str:
        metrics_string = ""
        for metric, details in self.metrics.items():
            metrics_string += f"Metric: {metric}\n"
            metrics_string += f"Definition: {details['description']}\nOptions:\n"
            for idx, (key, description) in enumerate(
                details["options"].items(), start=1
            ):
                metrics_string += f"  {idx}. {key}: {description}\n"
            metrics_string += "\n"
        return metrics_string.strip()

    def construct_json_schema(self) -> dict:
        json_schema = {
            "type": "object",
            "properties": {},
            "required": list(self.metrics.keys()) + ["justification"],
            # "additionalProperties": False,
        }

        for metric, details in self.metrics.items():
            options = details["options"]
            json_schema["properties"][metric] = {
                "type": "string",
                "enum": list(options.keys()),
                # "enumDescriptions": list(options.values()),
                "description": f"One of: {', '.join(options.keys())}",
            }

        json_schema["properties"]["justification"] = {
            "type": "object",
            "properties": {
                metric: {
                    "type": "string",
                    # "description": f"Explanation for the {metric} score.",
                }
                for metric in self.metrics
            },
            "required": list(self.metrics.keys()),
            # "additionalProperties": False,
        }

        print(json_schema)
        return json_schema


class PromptBuilder:
    def __init__(self, prompt_path: str):
        self.prompt_path = prompt_path
        self.prompts = self.load_prompts()

    def load_prompts(self) -> dict:
        try:
            with open(self.prompt_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML file: {e}")
            return {}

    def build(
        self,
        input_experience: str,
        job_description: str,
        output_experience: str,
        metrics_string: str,
    ):
        system_prompt_template = Template(
            self.prompts.get("system_prompt", "System prompt not found.")
        )
        user_prompt_template = Template(
            self.prompts.get("user_prompt", "User prompt not found.")
        )

        system_prompt = system_prompt_template.render(
            evaluation_metrics_string=metrics_string
        )
        user_prompt = user_prompt_template.render(
            input_experience=input_experience,
            job_description=job_description,
            output_experience=output_experience,
        )
        return system_prompt, user_prompt


class ResumeEvaluationEngine:
    # TODO: Extend ResumeEvaluationEngine to work with a CSV input for predictions, and a method to compare with ground truths.
    def __init__(self, metrics_path: str='./evaluation/metrics.json', prompt_path: str='./evaluation/prompts.yaml',eval_backend: str = "ollama",model=None):
        self.eval_backend = eval_backend
        self.model = model
        self.metric_processor = MetricProcessor(metrics_path)
        self.prompt_builder = PromptBuilder(prompt_path)
        self.evaluator = ResumeEvaluator(backend=self.eval_backend,model=self.model)

    def evaluate(
        self, input_experience: str, job_description: str, output_experience: str
    ):
        metrics_string = self.metric_processor.construct_metrics_string()
        system_prompt, user_prompt = self.prompt_builder.build(
            input_experience, job_description, output_experience, metrics_string
        )
        json_schema = self.metric_processor.construct_json_schema()
        return self.evaluator.evaluate(system_prompt, user_prompt, json_schema)

if __name__ == "__main__":
    input_exp = "Software Engineer with 5 years of experience in Python and Java."
    job_desc = "Looking for a Software Engineer with experience in Python, Java, and cloud technologies."
    output_exp = (
        "Software Engineer with 5 years of experience in Python, Java, and AWS."
    )

    engine = ResumeEvaluationEngine(
        metrics_path="./metrics.json", prompt_path="./prompts.yaml"
    )
    result = engine.evaluate(input_exp, job_desc, output_exp)

    from pprint import pprint

    pprint(result)
