from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, EmailStr, Field, HttpUrl

class WebsiteLinks(BaseModel):
    portfolio: Optional[HttpUrl]
    linkedIn: Optional[HttpUrl]
    github: Optional[HttpUrl]

class PersonalInfo(BaseModel):
    first_name: str
    middle_name: Optional[str]
    last_name: Optional[str]
    email: EmailStr
    contact_number: str
    summary: str
    websites: WebsiteLinks

class Education(BaseModel):
    title: str
    start_date: str
    end_date: str
    institute: str
    graduation: str
    score: str

class Skill(BaseModel):
    name: str
    tags: str

class Project(BaseModel):
    title: str
    organization: str
    start_date: str
    end_date: str
    link: Optional[HttpUrl]
    description: str
    tags: List[str]

class Experience(BaseModel):
    role: str
    company: str
    start_date: str
    end_date: str
    about_the_company: Optional[str] = Field(..., alias="about the company")
    location: str
    mode: Optional[str] = Literal["Remote","Onsite"]
    tags: List[str]
    description: str

class Certification(BaseModel):
    title: str
    organization: str
    link: Optional[HttpUrl]
    date: str
    tags: List[str]

class Achievement(BaseModel):
    award_title: str
    description: str

class MasterProfile(BaseModel):
    personal_info: PersonalInfo
    education: List[Education]
    skills: List[Skill]
    projects: List[Project]
    experience: List[Experience]
    certifications: List[Certification]
    achievements: List[Achievement]

class UserProfile(BaseModel):
    user_id: int
    master_profile: MasterProfile
    all_tags: List[str]

