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




