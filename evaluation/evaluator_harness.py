from evaluation.evaluator_base import ResumeEvaluationEngine
from dataclasses import dataclass
from typing import Callable
@dataclass
class ModelExperimentConfiguration:
    model_name: str
    prompt_path : str
    prediction_function: Callable[[str,str],str] # input experience, jd -> output experience

class ExperimentHarness(ResumeEvaluationEngine):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    

    
