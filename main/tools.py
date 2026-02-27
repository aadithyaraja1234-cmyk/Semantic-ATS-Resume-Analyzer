import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

IMPORTANT_PATTERNS = ["must have", "required", "mandatory", "strong experience", "expert"]

SKILL_CATEGORIES = {
    "Cloud": ["aws", "azure", "gcp", "cloud"],
    "DevOps": ["docker", "kubernetes", "terraform", "ci/cd"],
    "Backend": ["python", "java", "node", "spring"],
    "ML": ["machine learning", "deep learning", "tensorflow", "pytorch"],
    "Database": ["sql", "mysql", "postgresql", "mongodb"]
}


def clean_phrase(phrase):
    phrase = re.sub(r"^(a|an|the)\s+", "", phrase)
    return phrase.strip()


def extract_skills(text):
    doc = nlp(text.lower())
    skills = set()

    for chunk in doc.noun_chunks:
        phrase = clean_phrase(chunk.text)
        if len(phrase) > 2 and not phrase.isdigit():
            skills.add(phrase)

    return list(skills)


def detect_importance(jd_text, skill):
    jd_text = jd_text.lower()
    for pattern in IMPORTANT_PATTERNS:
        if pattern in jd_text and skill in jd_text:
            return 3
    return 1


def extract_weighted_skills(jd_text):
    skills = extract_skills(jd_text)
    weighted = {}
    for skill in skills:
        weighted[skill] = detect_importance(jd_text, skill)
    return weighted


def semantic_match(resume_skills, jd_weighted, threshold=0.6):
    if not resume_skills or not jd_weighted:
        return [], list(jd_weighted.keys())

    resume_emb = model.encode(resume_skills)
    jd_skills = list(jd_weighted.keys())
    jd_emb = model.encode(jd_skills)

    matched = []
    missing = []

    for i, jd_skill in enumerate(jd_skills):
        similarities = cosine_similarity(
            [jd_emb[i]], resume_emb
        )[0]

        if np.max(similarities) > threshold:
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

    for category, skills in SKILL_CATEGORIES.items():
        count = sum(1 for s in matched if any(k in s for k in skills))
        category_scores[category] = count

    return category_scores


def extract_years(text):
    match = re.search(r"(\d+)\s+years", text.lower())
    return int(match.group(1)) if match else 0


def detect_leadership(text):
    keywords = ["led", "managed", "mentored", "architected"]
    return any(word in text.lower() for word in keywords)


def detect_impact(text):
    return bool(re.findall(r"\d+%", text))


def compare_skills(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_weighted = extract_weighted_skills(jd_text)

    matched, missing = semantic_match(resume_skills, jd_weighted)
    score = calculate_weighted_score(matched, jd_weighted)

    category_scores = categorize_skills(matched)

    years = extract_years(resume_text)
    leadership = detect_leadership(resume_text)
    impact = detect_impact(resume_text)

    return {
        "matched": matched,
        "missing": missing,
        "score": score,
        "categories": category_scores,
        "years": years,
        "leadership": leadership,
        "impact": impact
    }