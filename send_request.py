import json
from pprint import pprint
import requests

user_data = {
    "master_profile": {
        "personal_info": {
            "title":"Dr",
            "first_name": "Aisha",
            "middle_name": "Zeenat",
            "last_name": "Iqbal",
            "summary": "Creative and resourceful BAJMC graduate with a passion for storytelling and mass communication in digital media.",
            "email": "aisha.iqbal@example.com",
            "contact_number": "+91 9765432108",
            "city":"Bangalore",
            "country":"India",
            "pin_code":"560029",
            "websites": [
                {"platform": "portfolio", "link": "https://aishaiqbal.com"},
                {"platform": "linkedIn", "link": "https://www.linkedin.com/in/aisha-iqbal"},
                {"platform": "github", "link": "https://github.com/aishaiqbal"},
            ],
        },
        "education": [
            {
                "title": "Bachelor of Arts in Journalism and Mass Communication",
                "institute": "Jamia Millia Islamia",
                "start_date": "2023-01-01",
                "graduation_date": "2026-01-01",
                "score": "8.0",
            },
            {
                "title": "Diploma in Digital Media",
                "institute": "Udemy",
                "start_date": "2022-06-01",
                "graduation_date": "2022-12-01",
                "score": None,
            },
        ],
        "skills": [
            {"name": "Journalism", "tags": ["writing", "reporting", "news"]},
            {"name": "Digital Media", "tags": ["campaigns", "social media", "marketing"]},
            {"name": "Editing", "tags": ["video editing", "audio editing"]},
            {"name": "Content Creation", "tags": ["content strategy", "copywriting"]},
        ],
        "projects": [
            {
                "title": "Social Media Campaign for NGO",
                "organization": "Self",
                "start_date": "2024-05-01",
                "end_date": "2024-07-01",
                "link": "https://aishaiqbal.com/ngo-campaign",
                "description": "Created and managed a social media campaign for a non-governmental organization to raise awareness about climate change.",
                "tags": ["digital media", "campaign", "social media"],
            },
            {
                "title": "Online News Platform Development",
                "organization": "Freelance",
                "start_date": "2023-11-01",
                "end_date": "2024-01-01",
                "link": "https://aishaiqbal.com/news-platform",
                "description": "Developed an online news platform using WordPress and integrated various digital media features.",
                "tags": ["website development", "digital media", "news"],
            },
        ],
        "experience": [
            {
                "role": "Journalism Intern",
                "company": "The Times of India",
                "location": "Delhi, India",
                "start_date": "2023-06-01",
                "end_date": "2023-08-01",
                "mode": "Onsite",
                "description": "Assisted in news reporting, writing articles, and conducting interviews for both print and online platforms.",
                "tags": ["journalism", "reporting", "writing"],
            },
            {
                "role": "Content Creator",
                "company": "Freelance",
                "location": "Remote",
                "start_date": "2023-02-01",
                "end_date": "2023-04-01",
                "mode": "Remote",
                "description": "Created content for various digital platforms, focusing on engaging with a younger audience.",
                "tags": ["content creation", "social media", "writing"],
            },
        ],
        "certifications": [
            {
                "title": "Digital Marketing Certification",
                "organization": "Google",
                "link": "https://google.com/certificate/abc123",
                "date": "2022-08-01",
                "tags": ["marketing", "digital"],
            },
            {
                "title": "Video Production Certification",
                "organization": "Coursera",
                "link": "https://coursera.org/certificate/video-production",
                "date": "2023-05-01",
                "tags": ["video production", "editing"],
            },
        ],
        "achievements": [
            {
                "award_title": "Best Journalism Internship",
                "description": "Awarded for excellence in reporting and writing at The Times of India internship program.",
                "date": "2023-08-01",
                "tags": ["reporting", "writing"],
            },
            {
                "award_title": "Top 10 Social Media Campaigns",
                "description": "Ranked in the top 10 best social media campaigns for environmental awareness in 2024.",
                "date": "2024-06-01",
                "tags": ["social media"],
            },
        ],
        "publications": [
            {
                "title": "IMPACT OF DIGITAL MEDIA ON SOCIETY",
                "publisher": "IEEE",
                "link": "https://ieee.com/2010",
                "date": "2024-05-01",
                "tags": ["digital media", "writing", "research"],
            }
        ],
    },
    "all_tags": [
        "journalism",
        "digital media",
        "writing",
        "reporting",
        "editing",
        "campaign",
        "social media",
        "website development",
        "news",
        "content creation",
        "marketing",
        "video production",
        "research",
    ],
}

job_description = """
Job Description:
Position: Digital Content Strategist
Company: GreenTech Media Solutions
Location: Remote

Requirements:
1. Strong background in digital content creation and campaign management.
2. Experience with social media marketing and audience engagement strategies.
3. Proficiency in video and audio editing tools.
4. Ability to develop content strategies and execute them across multiple digital platforms.
5. Prior experience working with NGOs or environmental causes is a plus.
6. Familiarity with WordPress and SEO best practices.

Responsibilities:
1. Plan and execute multi-channel content strategies.
2. Manage social media campaigns to increase engagement.
3. Create and edit multimedia content (videos, podcasts, etc.).
4. Collaborate
"""

selected_tags = [
    "campaign",
    "social media",
    "website development",
    "news",
    "content creation",
]

payload = {
    "llm_model":"smollm2",
    "user": user_data,
    "job_description": job_description,
    "selected_tags": selected_tags,
}

# API Call with Error Handling
try:
    response = requests.post(
        "http://localhost:8000/optimize_profile", json=payload, timeout=500
    )

    # Check if response status is OK (200)
    if response.status_code == 200:
        try:
            optimized_resume = response.json()
            print("Optimized Resume:", json.dumps(optimized_resume, indent=4))
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON response.")
    else:
        print(f"API Request failed with status code: {response.status_code}")
        print("Response:")
        pprint(response.json())

except requests.Timeout:
    print("Error: API request timed out.")
except requests.RequestException as e:
    print(f"Error: {e}")
