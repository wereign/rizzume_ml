from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from profile_processor import predict_on_master_profile

app = FastAPI()

# DATA MODELS
class InferenceData(BaseModel):
    user_id:int
    master_profile: dict
    all_tags: list

# ENDPOINTS
@app.get('/')
def read_root():
    return {"Message":"Home Page of the Inference Page"}


@app.post("/tag_profile")
def tag_profile(data:InferenceData):
    master_profile = data.master_profile
    all_tags = data.all_tags
    tagged_profile = predict_on_master_profile(master_profile,all_tags)

    return tagged_profile


# @app.post("/tag_profile")
# def tag_profile(data: InferenceData):
#     master_block = data.master_profile
#     all_tags = data.all_tags

#     # select the required blocks
#     blocks_list = ['projects', 'experience', 'certifications', 'achievements']

#     for block in blocks_list:
#         for i in range(len(master_block[block])):  # iterating over a list
#             block_data = master_block[block][i]
#             valid_keys = filter(lambda x: x != 'tags', block_data.keys())

#             # convert every block item into a string representation
#             x = str({i: block_data[i] for i in valid_keys})

#             # pass in the block with the user tags
#             predictions = predict(x, all_tags)
#             print('----------------------------------')
#             print(x)
#             print(predictions)

#             # set the predicted tags into the tags section of the item in the block
#     return {"place": "holder"}