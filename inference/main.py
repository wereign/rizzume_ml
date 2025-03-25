from typing import List, Union
from profile_model import UserProfile
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from profile_processor import predict_on_master_profile
from optimize import optimize_profile
from logger import Logger

logger = Logger(__name__).get_logger()
app = FastAPI()

class OptimizeModel(BaseModel):
    user : UserProfile
    job_description : str
    selected_tags : List[str]

# ENDPOINTS
@app.get('/')
def read_root():
    return {"Message":"Home Page of the Inference Service"}


@app.post("/tag_profile")
def tag_profile(data:UserProfile):
    print("called")
    master_profile = data.master_profile
    all_tags = data.all_tags
    tagged_profile = predict_on_master_profile(master_profile,all_tags)

    return tagged_profile


@app.post('/optimize_profile')
def optimize_profile_endpoint(payload:OptimizeModel):
    
    logger.info("Optimize Endpoint Invoked")
    logger.debug(payload)
    
    user_profile = payload.user
    profile = user_profile.master_profile
    job_description = payload.job_description
    selected_tags = payload.selected_tags

    # ensure that selected_tags are in all_tags
    for tag in selected_tags:
        if tag not in user_profile.all_tags:
            return HTTPException(status_code=400,details='Selected tags are not in the list of all tags created by the user')

    optimized_profile = optimize_profile(job_description=job_description,profile=profile,selected_tags=selected_tags)

    return optimized_profile
    