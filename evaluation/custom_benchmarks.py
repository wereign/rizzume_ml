from pydantic import BaseModel
from rubrics import faithfulness_rubric, alignment_rubric, conciseness_rubric,customization_rubric, grammar_rubric


def create_score_options(rubric,debug=False):
    strings = [f"{key}:{rubric[key]}" for key in rubric]
    if debug:
        print(strings)
    return 

faithfulness_rubric_options = create_score_options(faithfulness_rubric)
alignment_rubric_options = create_score_options(alignment_rubric)
conciseness_rubric_options = create_score_options(conciseness_rubric)
customization_rubric_options = create_score_options(customization_rubric)
grammar_rubric_options = create_score_options(grammar_rubric)