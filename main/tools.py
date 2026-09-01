import re
from sentence_transformers import SentenceTransformer, util

from skills_taxonomy import SKILL_TAXONOMY

_model = None


def get_embedding_model():
    """Lazily load the sentence-embedding model (only when semantic matching is needed)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


IMPORTANT_PATTERNS = ["must have", "required", "mandatory", "strong experience"]

# Derived from SKILL_TAXONOMY (single source of truth) --
#   SKILL_ALIASES: alias -> canonical name, e.g. "js" -> "javascript"
#   SKILL_CATEGORIES: category -> [canonical skill names in that category]
SKILL_ALIASES = {
    alias: canonical
    for canonical, info in SKILL_TAXONOMY.items()
    for alias in info["aliases"]
}

SKILL_CATEGORIES = {}
for _canonical, _info in SKILL_TAXONOMY.items():
    SKILL_CATEGORIES.setdefault(_info["category"], []).append(_canonical)

# Precompiled word-boundary-safe regex for every canonical name + alias, so
# "java" doesn't match inside "javascript" and "r" doesn't match inside
# "for". Built once at import time since this runs on every analysis.
_TERM_TO_CANONICAL = {canonical: canonical for canonical in SKILL_TAXONOMY}
_TERM_TO_CANONICAL.update(SKILL_ALIASES)

_SKILL_PATTERNS = [
    (re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"), canonical)
    for term, canonical in _TERM_TO_CANONICAL.items()
]


def canonicalize(term):
    return SKILL_ALIASES.get(term, term)


def find_taxonomy_skills(text):
    """Find which known skills (by canonical name or alias) appear in the text."""
    text_lower = text.lower()
    return {canonical for pattern, canonical in _SKILL_PATTERNS if pattern.search(text_lower)}


def split_sentences(text):
    """Plain punctuation-based sentence splitting -- no NLP model needed for this."""
    text = text.replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_skills(text):
    return list(find_taxonomy_skills(text))


def extract_weighted_skills(jd_text):
    """
    Find known skills in the JD and weight each one by whether it's
    mentioned in a sentence containing an importance signal (e.g.
    "required", "must have"), rather than just anywhere in the document.
    """
    weighted = {}

    for sentence in split_sentences(jd_text):
        weight = 3 if any(pattern in sentence.lower() for pattern in IMPORTANT_PATTERNS) else 1

        for skill in find_taxonomy_skills(sentence):
            # keep the strongest weight seen if a skill is mentioned more than once
            weighted[skill] = max(weighted.get(skill, 0), weight)

    return weighted


def semantic_match(resume_skills, jd_weighted, threshold=0.4):
    """
    JD skills are matched against resume skills exactly first (both sides
    are already canonical taxonomy names, so this is a direct set lookup,
    not a guess). Anything left over falls back to embedding similarity --
    safe here because both sides are clean curated skill names, not noisy
    extracted text.

    threshold=0.4 is the same empirical calibration used before this
    rewrite (see test_tools.py's TestSemanticMatchCalibration): the
    embedding vectors for a given pair of skill names don't change just
    because the surrounding extraction pipeline changed, so the earlier
    measurements are still valid evidence -- e.g. "postgresql"/"relational
    databases" (0.45) and "tensorflow"/"machine learning" (0.40) score
    above it, while "docker"/"kubernetes" (0.32, related tools but not the
    same skill) and "sql"/"javascript" (0.18) stay below it.
    """
    if not jd_weighted:
        return [], []

    resume_set = set(resume_skills)
    jd_skills = list(jd_weighted.keys())

    matched = [s for s in jd_skills if s in resume_set]
    remaining = [s for s in jd_skills if s not in resume_set]

    missing = []

    if remaining and resume_skills:
        model = get_embedding_model()
        resume_emb = model.encode(resume_skills, convert_to_tensor=True)
        jd_emb = model.encode(remaining, convert_to_tensor=True)
        scores = util.cos_sim(jd_emb, resume_emb)

        for i, skill in enumerate(remaining):
            if float(scores[i].max()) >= threshold:
                matched.append(skill)
            else:
                missing.append(skill)
    else:
        missing = remaining

    return matched, missing


def calculate_weighted_score(matched, jd_weighted):
    total_weight = sum(jd_weighted.values())
    matched_weight = sum(jd_weighted[s] for s in matched if s in jd_weighted)

    if total_weight == 0:
        return 0

    return round((matched_weight / total_weight) * 100, 2)


def calculate_confidence(score, jd_weighted, resume_skills):
    """
    How much to trust the match score, based on how much signal it's built
    on -- not a fixed fraction of the score itself. A 90% score derived from
    only 2 extracted requirements is less trustworthy than the same score
    derived from a dozen. Confidence ramps linearly with the combined number
    of JD + resume skill terms extracted, reaching full trust at 12 terms,
    then scales the score by that ramp.
    """
    signal_size = len(jd_weighted) + len(resume_skills)
    signal_factor = min(1.0, signal_size / 12)
    return round(score * signal_factor, 2)


def categorize_skills(matched):
    """
    matched entries are already exact canonical taxonomy names (see
    semantic_match), so this is a direct lookup, not substring guessing.
    """
    category_scores = {category: 0 for category in SKILL_CATEGORIES}

    for skill in matched:
        for category, skills in SKILL_CATEGORIES.items():
            if skill in skills:
                category_scores[category] += 1
                break  # each taxonomy skill belongs to exactly one category

    return category_scores


LEADERSHIP_KEYWORDS = [
    "led", "lead ", "leadership", "managed", "manager", "mentored",
    "mentor", "supervised", "supervisor", "directed", "director",
    "head of", "coordinated", "spearheaded"
]


def extract_years_experience(resume_text):
    matches = re.findall(
        r"(\d+)\+?\s*(?:years|yrs)\.?\s*(?:of)?\s*experience",
        resume_text.lower()
    )
    if not matches:
        return 0
    return max(int(m) for m in matches)


def detect_leadership(resume_text):
    text = resume_text.lower()
    return any(keyword in text for keyword in LEADERSHIP_KEYWORDS)


def detect_impact_metrics(resume_text):
    return bool(re.search(r"\d+%|\$\d+[kKmMbB]?|\b\d+x\b", resume_text))


def compare_skills(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_weighted = extract_weighted_skills(jd_text)

    matched, missing = semantic_match(resume_skills, jd_weighted)
    score = calculate_weighted_score(matched, jd_weighted)
    confidence = calculate_confidence(score, jd_weighted, resume_skills)

    categories = categorize_skills(matched)

    return {
        "matched": matched,
        "missing": missing,
        "score": score,
        "confidence": confidence,
        "categories": categories,
        "years": extract_years_experience(resume_text),
        "leadership": detect_leadership(resume_text),
        "impact": detect_impact_metrics(resume_text)
    }
