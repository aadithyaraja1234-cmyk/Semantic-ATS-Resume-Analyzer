"""
Curated skill taxonomy: canonical skill name -> category + known aliases.

This is the single source of truth for skill extraction, alias
canonicalization, and category distribution -- previously those were three
separate, inconsistent mechanisms (spaCy noun-chunk extraction, a tiny
5-category dict, and a small unrelated alias table). Matching against a
curated vocabulary instead of freely extracting noun phrases is also just
how real keyword-based ATS tools work, and it eliminates an entire class of
junk results (noun-chunk extraction was picking up things like "35%",
"improved latency", and "senior backend engineer" as if they were skills).
"""

SKILL_TAXONOMY = {
    # Programming Languages
    "python": {"category": "Programming Languages", "aliases": []},
    "java": {"category": "Programming Languages", "aliases": []},
    "javascript": {"category": "Programming Languages", "aliases": ["js"]},
    "typescript": {"category": "Programming Languages", "aliases": ["ts"]},
    "c++": {"category": "Programming Languages", "aliases": ["cpp"]},
    "c#": {"category": "Programming Languages", "aliases": ["csharp"]},
    "go": {"category": "Programming Languages", "aliases": ["golang"]},
    "rust": {"category": "Programming Languages", "aliases": []},
    "ruby": {"category": "Programming Languages", "aliases": []},
    "php": {"category": "Programming Languages", "aliases": []},
    "swift": {"category": "Programming Languages", "aliases": []},
    "kotlin": {"category": "Programming Languages", "aliases": []},
    "scala": {"category": "Programming Languages", "aliases": []},
    "matlab": {"category": "Programming Languages", "aliases": []},
    "dart": {"category": "Programming Languages", "aliases": []},

    # Frontend Development
    "react": {"category": "Frontend Development", "aliases": ["react.js", "reactjs"]},
    "angular": {"category": "Frontend Development", "aliases": []},
    "vue": {"category": "Frontend Development", "aliases": ["vue.js", "vuejs"]},
    "next.js": {"category": "Frontend Development", "aliases": ["nextjs"]},
    "svelte": {"category": "Frontend Development", "aliases": []},
    "html": {"category": "Frontend Development", "aliases": ["html5"]},
    "css": {"category": "Frontend Development", "aliases": ["css3"]},
    "sass": {"category": "Frontend Development", "aliases": ["scss"]},
    "tailwind css": {"category": "Frontend Development", "aliases": ["tailwind", "tailwindcss"]},
    "redux": {"category": "Frontend Development", "aliases": []},
    "webpack": {"category": "Frontend Development", "aliases": []},
    "responsive design": {"category": "Frontend Development", "aliases": []},

    # Backend Development
    "node.js": {"category": "Backend Development", "aliases": ["node", "nodejs"]},
    "express.js": {"category": "Backend Development", "aliases": ["express", "expressjs"]},
    "django": {"category": "Backend Development", "aliases": []},
    "flask": {"category": "Backend Development", "aliases": []},
    "fastapi": {"category": "Backend Development", "aliases": []},
    "spring boot": {"category": "Backend Development", "aliases": ["spring"]},
    "ruby on rails": {"category": "Backend Development", "aliases": ["rails"]},
    "asp.net": {"category": "Backend Development", "aliases": [".net", "dotnet"]},
    "graphql": {"category": "Backend Development", "aliases": []},
    "rest api": {"category": "Backend Development", "aliases": ["restful api", "rest apis"]},
    "grpc": {"category": "Backend Development", "aliases": []},
    "microservices": {"category": "Backend Development", "aliases": []},

    # Mobile Development
    "ios development": {"category": "Mobile Development", "aliases": ["ios"]},
    "android development": {"category": "Mobile Development", "aliases": ["android"]},
    "react native": {"category": "Mobile Development", "aliases": []},
    "flutter": {"category": "Mobile Development", "aliases": []},
    "xamarin": {"category": "Mobile Development", "aliases": []},

    # Cloud & Infrastructure
    "amazon web services": {"category": "Cloud & Infrastructure", "aliases": ["aws"]},
    "microsoft azure": {"category": "Cloud & Infrastructure", "aliases": ["azure"]},
    "google cloud platform": {"category": "Cloud & Infrastructure", "aliases": ["gcp", "google cloud"]},
    "cloud infrastructure": {"category": "Cloud & Infrastructure", "aliases": ["cloud computing"]},
    "serverless": {"category": "Cloud & Infrastructure", "aliases": []},
    "aws lambda": {"category": "Cloud & Infrastructure", "aliases": ["lambda"]},
    "ec2": {"category": "Cloud & Infrastructure", "aliases": []},
    "s3": {"category": "Cloud & Infrastructure", "aliases": []},
    "cloudformation": {"category": "Cloud & Infrastructure", "aliases": []},

    # DevOps & CI/CD
    "docker": {"category": "DevOps & CI/CD", "aliases": ["containerization"]},
    "kubernetes": {"category": "DevOps & CI/CD", "aliases": ["k8s"]},
    "terraform": {"category": "DevOps & CI/CD", "aliases": ["infrastructure as code", "iac"]},
    "ansible": {"category": "DevOps & CI/CD", "aliases": []},
    "jenkins": {"category": "DevOps & CI/CD", "aliases": []},
    "github actions": {"category": "DevOps & CI/CD", "aliases": []},
    "gitlab ci": {"category": "DevOps & CI/CD", "aliases": ["gitlab ci/cd"]},
    "continuous integration and continuous deployment": {"category": "DevOps & CI/CD", "aliases": ["ci/cd", "cicd"]},
    "helm": {"category": "DevOps & CI/CD", "aliases": []},
    "prometheus": {"category": "DevOps & CI/CD", "aliases": []},
    "grafana": {"category": "DevOps & CI/CD", "aliases": []},

    # Databases
    "sql": {"category": "Databases", "aliases": []},
    "mysql": {"category": "Databases", "aliases": []},
    "postgresql": {"category": "Databases", "aliases": ["postgres"]},
    "mongodb": {"category": "Databases", "aliases": ["mongo"]},
    "redis": {"category": "Databases", "aliases": []},
    "elasticsearch": {"category": "Databases", "aliases": []},
    "cassandra": {"category": "Databases", "aliases": []},
    "dynamodb": {"category": "Databases", "aliases": []},
    "sqlite": {"category": "Databases", "aliases": []},
    "relational databases": {"category": "Databases", "aliases": []},
    "nosql databases": {"category": "Databases", "aliases": ["nosql"]},
    "database design": {"category": "Databases", "aliases": []},

    # Data Science & Machine Learning
    "machine learning": {"category": "Data Science & Machine Learning", "aliases": ["ml"]},
    "deep learning": {"category": "Data Science & Machine Learning", "aliases": ["dl"]},
    "tensorflow": {"category": "Data Science & Machine Learning", "aliases": []},
    "pytorch": {"category": "Data Science & Machine Learning", "aliases": []},
    "scikit-learn": {"category": "Data Science & Machine Learning", "aliases": ["sklearn"]},
    "pandas": {"category": "Data Science & Machine Learning", "aliases": []},
    "numpy": {"category": "Data Science & Machine Learning", "aliases": []},
    "natural language processing": {"category": "Data Science & Machine Learning", "aliases": ["nlp"]},
    "computer vision": {"category": "Data Science & Machine Learning", "aliases": []},
    "data science": {"category": "Data Science & Machine Learning", "aliases": []},
    "data engineering": {"category": "Data Science & Machine Learning", "aliases": []},
    "apache spark": {"category": "Data Science & Machine Learning", "aliases": ["spark", "pyspark"]},
    "hadoop": {"category": "Data Science & Machine Learning", "aliases": []},
    "apache airflow": {"category": "Data Science & Machine Learning", "aliases": ["airflow"]},
    "large language models": {"category": "Data Science & Machine Learning", "aliases": ["llm", "llms"]},
    "generative ai": {"category": "Data Science & Machine Learning", "aliases": ["genai"]},
    "statistics": {"category": "Data Science & Machine Learning", "aliases": []},

    # Testing & QA
    "unit testing": {"category": "Testing & QA", "aliases": []},
    "integration testing": {"category": "Testing & QA", "aliases": []},
    "selenium": {"category": "Testing & QA", "aliases": []},
    "cypress": {"category": "Testing & QA", "aliases": []},
    "jest": {"category": "Testing & QA", "aliases": []},
    "pytest": {"category": "Testing & QA", "aliases": []},
    "test automation": {"category": "Testing & QA", "aliases": []},
    "quality assurance": {"category": "Testing & QA", "aliases": ["qa"]},
    "test driven development": {"category": "Testing & QA", "aliases": ["tdd"]},

    # Security
    "cybersecurity": {"category": "Security", "aliases": ["cyber security"]},
    "penetration testing": {"category": "Security", "aliases": ["pen testing"]},
    "encryption": {"category": "Security", "aliases": []},
    "oauth": {"category": "Security", "aliases": []},
    "identity and access management": {"category": "Security", "aliases": ["iam"]},
    "compliance": {"category": "Security", "aliases": []},
    "vulnerability assessment": {"category": "Security", "aliases": []},

    # Version Control & Collaboration
    "git": {"category": "Version Control & Collaboration", "aliases": []},
    "github": {"category": "Version Control & Collaboration", "aliases": []},
    "gitlab": {"category": "Version Control & Collaboration", "aliases": []},
    "bitbucket": {"category": "Version Control & Collaboration", "aliases": []},
    "jira": {"category": "Version Control & Collaboration", "aliases": []},
    "confluence": {"category": "Version Control & Collaboration", "aliases": []},

    # Design & UX
    "figma": {"category": "Design & UX", "aliases": []},
    "sketch": {"category": "Design & UX", "aliases": []},
    "adobe xd": {"category": "Design & UX", "aliases": []},
    "user interface design": {"category": "Design & UX", "aliases": ["ui"]},
    "user experience design": {"category": "Design & UX", "aliases": ["ux"]},
    "wireframing": {"category": "Design & UX", "aliases": []},
    "prototyping": {"category": "Design & UX", "aliases": []},

    # Project Management & Methodologies
    "agile": {"category": "Project Management & Methodologies", "aliases": []},
    "scrum": {"category": "Project Management & Methodologies", "aliases": []},
    "kanban": {"category": "Project Management & Methodologies", "aliases": []},
    "project management": {"category": "Project Management & Methodologies", "aliases": []},
    "product management": {"category": "Project Management & Methodologies", "aliases": []},
    "stakeholder management": {"category": "Project Management & Methodologies", "aliases": []},

    # Analytics & BI
    "tableau": {"category": "Analytics & BI", "aliases": []},
    "power bi": {"category": "Analytics & BI", "aliases": ["powerbi"]},
    "microsoft excel": {"category": "Analytics & BI", "aliases": ["excel"]},
    "google analytics": {"category": "Analytics & BI", "aliases": []},
    "data visualization": {"category": "Analytics & BI", "aliases": []},

    # Soft Skills -- resumes describe these as actions ("led a team",
    # "mentored engineers"), not the noun form, so common verb inflections
    # are included as aliases alongside the noun form itself.
    "leadership": {"category": "Soft Skills", "aliases": ["led a team", "leading a team"]},
    "communication": {"category": "Soft Skills", "aliases": ["communicated", "communicating"]},
    "teamwork": {"category": "Soft Skills", "aliases": ["collaboration", "collaborated", "collaborating"]},
    "problem solving": {"category": "Soft Skills", "aliases": ["problem-solving"]},
    "mentoring": {"category": "Soft Skills", "aliases": ["mentorship", "mentored"]},
    "critical thinking": {"category": "Soft Skills", "aliases": []},
    "time management": {"category": "Soft Skills", "aliases": []},
}
