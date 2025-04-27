from icecream import ic
from datetime import datetime
from evaluation.evaluator_base import ResumeEvaluationEngine
import pandas as pd

OUTPUT_DIR = "./eval_test"

class EvaluatorTester(ResumeEvaluationEngine):
    def evaluate_from_csv(self, csv_path, output_dir_path: str = OUTPUT_DIR):
        original_df = pd.read_csv(csv_path)
        og_columns = list(original_df.columns)

        # Create a new DataFrame with additional prediction columns
        output_columns = og_columns + [
            "pred_factual_accuracy",
            "pred_alignment",
            "pred_section_length",
            "pred_grammar",
            "pred_justification",
        ]
        eval_df = original_df.copy()
        for col in output_columns:
            if col not in eval_df.columns:
                eval_df[col] = None  # Initialize new columns

        # Iterate through and evaluate each row
        for i, row in eval_df.iterrows():
            result = self.evaluate(
                row["input_experience"],
                row["job_description"],
                row["output_experience"],
            )
            ic(result)
            for key in result:
                df_key = f"pred_{key}"
                eval_df.at[i, df_key] = str(result[key])

        # Write evaluated DataFrame to output file
        time_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        output_path = (
            f"{output_dir_path}/eval_{time_string}_{self.eval_backend}_{self.model}.csv"
        )
        ic(output_path)
        eval_df.to_csv(output_path, index=False)