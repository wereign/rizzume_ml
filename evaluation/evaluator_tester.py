import os
import time
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
        output_columns = og_columns + [f"pred_{i}" for i in self.metric_names]
        
        eval_df = original_df.copy()
        for col in output_columns:
            if col not in eval_df.columns:
                eval_df[col] = None  # Initialize new columns

        # Iterate through and evaluate each row
        for i, row in eval_df.iterrows():
            print(f"Evaluating Sample: {i}")
            print(row["input_experience"][:50])
            print(row["job_description"][:50])
            print(row["output_experience"][:50])
            print()

            if i + 1 % 15 == 0:
                time.sleep(30)
                
            result = self.evaluate(
                input_experience = row["input_experience"],
                job_description=row["job_description"],
                output_experience=row["output_experience"],
            )
            for key in result:
                df_key = f"pred_{key}"
                eval_df.at[i, df_key] = str(result[key])

        # Write evaluated DataFrame to output file
        time_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        output_path = (
            f"{output_dir_path}/eval_{time_string}_{self.eval_backend}_{self.model}.csv"
        )
        # ic(output_path)
        eval_df.to_csv(output_path, index=False)
        
        if not hasattr(self,"_eval_df"):
            self._eval_df = eval_df

        return eval_df  

    def analyze_results(self,csv_path=None):
        if not hasattr(self,"_eval_df") and not csv_path:
            raise AttributeError("Please run evaluation first or provide path to saved csv")
        # if both, csv path and the attribute exist and are passed as arguments, then 
        elif csv_path:
            self._eval_df = pd.read_csv(csv_path)
        elif hasattr(self,'_eval_df'):
            pass



