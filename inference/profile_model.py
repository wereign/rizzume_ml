from typing import List, Literal, Optional
from pydantic import BaseModel, EmailStr, Field, HttpUrl
from datetime import datetime

class WebsiteLinks(BaseModel):
    platform: str
    link: Optional[HttpUrl]

class PersonalInfo(BaseModel):
    first_name: str
    middle_name: Optional[str]
    last_name: Optional[str]
    email: EmailStr
    contact_number: str
    summary: str
    websites: Optional[List[WebsiteLinks]]

class Education(BaseModel):
    title: str
    institute: str
    start_date: str
    graduation_date: Optional[str]
    score: Optional[str]

class Skill(BaseModel):
    name: str

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
    location: str
    mode: Optional[str] = Literal["Remote","Onsite"]
    tags: List[str]
    description: str

class Certification(BaseModel):
    title: str
    organization: str
    link: Optional[HttpUrl]
    date: Optional[datetime]
    tags: List[str]

class Achievement(BaseModel):
    award_title: str
    description: str
    date: Optional[str]


class Publications(BaseModel):
    title: str
    publisher: str
    link : Optional[HttpUrl]
    date: str
    tags: List[str]

class MasterProfile(BaseModel):
    personal_info: PersonalInfo
    education: List[Education]
    skills: List[Skill]
    projects: List[Project]
    experience: List[Experience]
    certifications: Optional[List[Certification]]
    achievements: Optional[List[Achievement]]
    publications: Optional[List[Publications]]
class UserProfile(BaseModel):
    user_id: str
    master_profile: MasterProfile
    all_tags: List[str]

