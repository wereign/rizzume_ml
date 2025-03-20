from typing import List, Union
from profile_model import UserProfile
from pydantic import BaseModel
from fastapi import FastAPI
from profile_processor import predict_on_master_profile
from optimize import optimize_profile
app = FastAPI()


class OptimizeModel(BaseModel):
    user : UserProfile
    job_description : str



# ENDPOINTS
@app.get('/')
def read_root():
    return {"Message":"Home Page of the Inference Page"}


@app.post("/tag_profile")
def tag_profile(data:UserProfile):
    print("called")
    master_profile = data.master_profile
    all_tags = data.all_tags
    tagged_profile = predict_on_master_profile(master_profile,all_tags)

    return tagged_profile


@app.post('/optimize_profile')
def optimize_profile_endpoint(data:OptimizeModel):
    print("Got Optimization Request")
    user = data.user
    profile = user.master_profile

    job_description = data.job_description
    optimized_profile = optimize_profile(job_description=job_description,profile_json=profile)

    return optimized_profile
    