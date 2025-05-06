import os
import time
from icecream import ic
from datetime import datetime
from evaluation.evaluator_base import ResumeEvaluationEngine
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "./eval_test"

class EvaluatorTester(ResumeEvaluationEngine):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self.pred_columns = [f"pred_{i}" for i in self.metric_names]

    def evaluate_from_csv(self, csv_path, output_dir_path: str = OUTPUT_DIR):
        original_df = pd.read_csv(csv_path)
        og_columns = list(original_df.columns)

        # Create a new DataFrame with additional prediction columns
        output_columns = og_columns + [f"pred_{i}" for i in self.metric_names]
        output_columns = og_columns + self.pred_columns

        eval_df = original_df.copy()

        for col in output_columns:
            if col not in eval_df.columns:
                eval_df[col] = None  # Initialize new columns

         # Iterate through and evaluate each row
        for i, row in eval_df.iterrows():
            print()
            print(f"Evaluating Sample: {i}")
            # print(row["input_experience"][:50])
            # print(row["job_description"][:50])
            # print(row["output_experience"][:50])

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


    def analyze_results(self, csv_path=None):
        if not (hasattr(self, "_eval_df") and hasattr(self, "_csv_path")) and not csv_path:
            raise AttributeError("Please run evaluation first or provide path to saved csv")
        elif csv_path:
            self._eval_df = pd.read_csv(csv_path)
            self._csv_path = csv_path
        elif hasattr(self, "_eval_df") and hasattr(self, "_csv_path"):
            csv_path = self._csv_path

        # Create output directory from CSV file name
        csv_dir = os.path.dirname(csv_path)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        plot_dir = os.path.join(csv_dir, base_name)
        os.makedirs(plot_dir, exist_ok=True)

        label_pred_col_pairs = list(zip(self.metric_names, self.pred_columns))
        analysis_dict = {}

        for label_col, pred_col in label_pred_col_pairs:
            labels = self._eval_df[label_col]
            preds = self._eval_df[pred_col]
            cr = classification_report(labels, preds)
            print(f"\n--- {label_col} vs {pred_col} ---\n")
            print(cr)

            analysis_dict[label_col] = cr

            unique_classes = sorted(set(labels.unique()).union(preds.unique()))
            correct_counts = []
            incorrect_counts = []

            for cls in unique_classes:
                cls_mask = (labels == cls)
                correct = ((preds == cls) & cls_mask).sum()
                incorrect = (cls_mask & (preds != cls)).sum()
                correct_counts.append(correct)
                incorrect_counts.append(incorrect)

            # Plotting per-class histogram
            x = range(len(unique_classes))
            plt.figure(figsize=(8, 5))
            plt.bar(x, correct_counts, color='blue', label='Correct')
            plt.bar(x, incorrect_counts, bottom=correct_counts, color='red', label='Incorrect')
            plt.xticks(x, unique_classes)
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.title(f'Class-wise Prediction Accuracy\n{label_col} vs {pred_col}')
            plt.legend()
            plt.tight_layout()

            # Save plot to file
            plot_path = os.path.join(plot_dir, f"{label_col}_vs_{pred_col}_accuracy.png")
            plt.savefig(plot_path)
            plt.close()  # Close the figure to free memory

        return analysis_dict




