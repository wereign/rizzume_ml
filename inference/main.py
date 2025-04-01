import json
from typing import List, Literal
from profile_model import MasterProfile, UserProfile
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from profile_processor import predict_on_master_profile
from optimize import optimize_profile
from logger import Logger

logger = Logger(__name__).get_logger()
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify specific domains: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class OptimizeModel(BaseModel):
    master_profile : MasterProfile
    llm_model: Literal['llama3.1','llama3.2','smollm2','gemma3:4b','gemma3:1b','command-r7b']
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
    
    
    profile = payload.master_profile
    job_description = payload.job_description
    selected_tags = payload.selected_tags
    model = payload.llm_model

    # # ensure that selected_tags are in all_tags
    # for tag in selected_tags:
    #     if tag not in user_profile.all_tags:
    #         return HTTPException(status_code=400,details='Selected tags are not in the list of all tags created by the user')
    
    logger.info("Optimizing")
    optimized_profile = optimize_profile(model_name=model,job_description=job_description,profile=profile,selected_tags=selected_tags)
    return json.loads(optimized_profile)
    