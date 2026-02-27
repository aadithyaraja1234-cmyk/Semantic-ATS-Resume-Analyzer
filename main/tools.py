import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

IMPORTANT_PATTERNS = ["must have", "required", "mandatory", "strong experience"]

SKILL_CATEGORIES = {
    "Cloud": ["aws", "azure", "gcp", "cloud"],
    "DevOps": ["docker", "kubernetes", "terraform", "ci/cd"],
    "Backend": ["python", "java", "node"],
    "ML": ["machine learning", "deep learning", "tensorflow"],
    "Database": ["sql", "mysql", "postgresql"]
}


def clean_phrase(phrase):
    phrase = re.sub(r"^(a|an|the)\s+", "", phrase)
    return phrase.strip()


def extract_skills(text):
    doc = nlp(text.lower())
    skills = set()

    for chunk in doc.noun_chunks:
        phrase = clean_phrase(chunk.text)
        if len(phrase) > 2:
            skills.add(phrase)

    return list(skills)


def detect_importance(jd_text, skill):
    for pattern in IMPORTANT_PATTERNS:
        if pattern in jd_text.lower() and skill in jd_text.lower():
            return 3
    return 1


def extract_weighted_skills(jd_text):
    skills = extract_skills(jd_text)
    return {skill: detect_importance(jd_text, skill) for skill in skills}


def semantic_match(resume_skills, jd_weighted, threshold=0.2):
    if not resume_skills or not jd_weighted:
        return [], list(jd_weighted.keys())

    vectorizer = TfidfVectorizer().fit(resume_skills + list(jd_weighted.keys()))

    resume_vec = vectorizer.transform(resume_skills)
    jd_vec = vectorizer.transform(list(jd_weighted.keys()))

    matched = []
    missing = []

    for i, jd_skill in enumerate(jd_weighted.keys()):
        sim_scores = cosine_similarity(jd_vec[i], resume_vec)[0]
        if max(sim_scores) > threshold:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return matched, missing


def calculate_weighted_score(matched, jd_weighted):
    total_weight = sum(jd_weighted.values())
    matched_weight = sum(jd_weighted[s] for s in matched if s in jd_weighted)

    if total_weight == 0:
        return 0

    return round((matched_weight / total_weight) * 100, 2)


def categorize_skills(matched):
    category_scores = {}

    for category, keywords in SKILL_CATEGORIES.items():
        count = sum(1 for s in matched if any(k in s.lower() for k in keywords))
        category_scores[category] = count

    return category_scores


def compare_skills(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_weighted = extract_weighted_skills(jd_text)

    matched, missing = semantic_match(resume_skills, jd_weighted)
    score = calculate_weighted_score(matched, jd_weighted)

    categories = categorize_skills(matched)

    return {
        "matched": matched,
        "missing": missing,
        "score": score,
        "categories": categories
    }
