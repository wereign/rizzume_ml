import json
from pprint import pprint
import requests

user_data = {
    "master_profile": {
        "personal_info": {
            "first_name": "Emily",
            "middle_name": "Grace",
            "last_name": "Johnson",
            "summary": "Passionate product designer with experience in UI/UX design, prototyping, and user research.",
            "email": "emily.johnson@example.com",
            "contact_number": "+1 987 654 3210",
            "websites": [
                {"platform": "portfolio", "link": "https://emilyjohnson.com"},
                {
                    "platform": "linkedIn",
                    "link": "https://www.linkedin.com/in/emily-johnson",
                },
                {"platform": "github", "link": "https://github.com/emilyjohnson"},
            ],
            "city": "New York",
            "title": "Product Designer",
            "country": "USA",
            "pin_code": "07008",
        },
        "education": [
            {
                "title": "Bachelor of Fine Arts in Graphic Design",
                "institute": "Parsons School of Design",
                "start_date": "2019-09-01T00:00:00.000Z",
                "graduation_date": "2023-06-01T00:00:00.000Z",
                "score": "3.7",
            },
            {
                "title": "UI/UX Design Bootcamp",
                "institute": "General Assembly",
                "start_date": "2022-01-01T00:00:00.000Z",
                "graduation_date":  "2022-04-01T00:00:00.000Z",
                "score": None,
            },
        ],
        "skills": [
            {"name": "UI/UX Design"},
            {"name": "Graphic Design"},
            {"name": "Wireframing"},
            {"name": "User Testing"},
        ],
        "projects": [
            {
                "title": "Mobile App Redesign for E-commerce",
                "organization": "Freelance",
                "start_date": "2023-02-01T00:00:00.000Z",
                "end_date":  "2023-04-01T00:00:00.000Z",
                "link": "https://emilyjohnson.com/mobile-app-redesign",
                "description": "Redesigned an e-commerce mobile app for improved user experience and aesthetics.",
                "tags": ["UI/UX design", "mobile app", "e-commerce"],
            },
            {
                "title": "Website Redesign for Local Bakery",
                "organization": "Self",
                "start_date": "2023-05-01T00:00:00.000Z",
                "end_date": "2023-07-01T00:00:00.000Z",
                "link": "https://emilyjohnson.com/bakery-website",
                "description": "Created a responsive and visually appealing website for a local bakery.",
                "tags": ["web design", "UI/UX", "branding"],
            },
        ],
        "experience": [
            {
                "role": "UI/UX Designer Intern",
                "company": "Design Studio",
                "location": "New York, USA",
                "start_date": "2022-06-01T00:00:00.000Z",
                "end_date": "2022-08-01T00:00:00.000Z",
                "mode": "Onsite",
                "description": "Worked on prototyping, user research, and UI design for multiple digital products.",
                "tags": ["UI/UX design", "user research", "prototyping"],
            },
            {
                "role": "Freelance Graphic Designer",
                "company": "Self",
                "location": "Remote",
                "start_date": "2022-09-01T00:00:00.000Z",
                "end_date": "2023-01-01T00:00:00.000Z",
                "mode": "Remote",
                "description": "Designed logos, websites, and marketing materials for clients across industries.",
                "tags": ["graphic design", "branding", "logos"],
            },
        ],
        "certifications": [
            {
                "title": "UI/UX Design Certification",
                "organization": "General Assembly",
                "link": "https://generalassembly.com/certificate/ux-design",
                "date": "2022-04-01T00:00:00.000Z",
                "tags": ["design", "prototyping"],
            }
        ],
        "achievements": [
            {
                "award_title": "Best Product Design",
                "description": "Won the award for best product design in a national competition in 2023.",
                "date": "2023-06-01T00:00:00.000Z",
                "tags": ["design", "product design", "award"],
            }
        ],
        "publications": [
            {
                "title": "Designing the Future: UI/UX in 2025",
                "publisher": "Design Trends Journal",
                "link": "https://designtrendsjournal.com/designing-the-future",
                "date": "2023-06-01T00:00:00.000Z",
                "tags": ["UI/UX design", "future trends", "product design"],
            },
            {
                "title": "Prototyping for User-Centered Design",
                "publisher": "UX Design Weekly",
                "link": "https://uxdesignweekly.com/prototyping-for-user-centered-design",
                "date": "2022-11-01T00:00:00.000Z",
                "tags": ["prototyping", "UI/UX design", "user-centered design"],
            },
        ],
    },
    "all_tags": [
        "UI/UX design",
        "graphic design",
        "wireframing",
        "user testing",
        "prototyping",
        "user research",
        "branding",
        "Photoshop",
        "Illustrator",
        "Sketch",
        "Figma",
        "InVision",
        "future trends",
        "product design",
        "user-centered design",
    ],
}

job_description = """
Job Description:
Position: UI/UX Designer
Company: Innovative Design Solutions
Location: Remote / New York, USA

Requirements:
Strong background in UI/UX design, graphic design, and user research.
Proficiency in design tools like Figma, Sketch, Photoshop, Illustrator, and InVision.
Experience in wireframing, prototyping, and user testing.
Ability to conduct user research and translate insights into intuitive design solutions.
Strong portfolio showcasing mobile and web design projects.
Experience in branding and digital product aesthetics is a plus.

Responsibilities:

Design and prototype user-friendly interfaces for web and mobile applications.
Conduct user research, usability testing, and gather feedback to improve designs.
Develop wireframes, user flows, and high-fidelity UI mockups.
Collaborate with developers, product managers, and stakeholders to refine design solutions.
Ensure consistency in branding and user experience across digital products.
Stay updated on the latest UI/UX trends and best practices.

"""
selected_tags = [
    "Figma",
    "InVision",
    "future trends",
    "product design",
    "user-centered design",
]

payload = {
    "llm_model": "smollm2",
    "user": user_data,
    "job_description": job_description,
    "selected_tags": selected_tags,
}

# API Call with Error Handling
try:
    response = requests.post(
        "http://localhost:8080/optimize_profile", json=payload, timeout=500
    )

    # Check if response status is OK (200)
    if response.status_code == 200:
        try:
            optimized_resume = response.json()
            print("Response:")
            print(optimized_resume.keys())
            print(type(optimized_resume))
            pprint(optimized_resume)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON response.")
    else:
        print(f"API Request failed with status code: {response.status_code}")

except requests.Timeout:
    print("Error: API request timed out.")
except requests.RequestException as e:
    print(f"Error: {e}")
