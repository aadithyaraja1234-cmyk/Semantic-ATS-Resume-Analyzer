import re
import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")

_model = None


def get_embedding_model():
    """Lazily load the sentence-embedding model (only when semantic matching is needed)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


IMPORTANT_PATTERNS = ["must have", "required", "mandatory", "strong experience"]

SKILL_CATEGORIES = {
    "Cloud": ["aws", "azure", "gcp", "cloud"],
    "DevOps": ["docker", "kubernetes", "terraform", "ci/cd"],
    "Backend": ["python", "java", "node"],
    "ML": ["machine learning", "deep learning", "tensorflow"],
    "Database": ["sql", "mysql", "postgresql"]
}

# Generic noun phrases that show up constantly in resumes/JDs but are never
# actual skills (e.g. "years of experience", "our team"). Filtering these
# out keeps matched/missing lists focused on real skill terms. Checked AFTER
# clean_phrase(), so both "candidate" and "a candidate" are caught.
GENERIC_TERMS = {
    "years", "year", "experience", "years of experience", "team", "teams",
    "company", "companies", "role", "roles", "position", "positions",
    "candidate", "candidates", "opportunity", "opportunities", "environment",
    "ability", "abilities", "knowledge", "work", "responsibility",
    "responsibilities", "skill", "skills", "job", "jobs", "background",
    "understanding", "plus", "someone", "individual", "who", "what", "this",
    "that", "it", "we", "our", "us", "you", "they"
}

# Qualifier words that precede a real skill/requirement but aren't part of
# it (e.g. "strong experience" -> "experience", "solid understanding" ->
# "understanding" -- both then dropped by GENERIC_TERMS).
QUALIFIER_PREFIXES = ("strong ", "solid ", "excellent ", "proven ", "good ", "extensive ")

# spaCy's noun_chunks doesn't split conjunctions ("python and cloud
# infrastructure" comes back as one chunk), which buries individual skills
# inside compound phrases. Splitting on these delimiters recovers them.
CONJUNCT_SPLIT = re.compile(r"\s*(?:,|/|&|\band\b|\bor\b)\s*")

# Common abbreviations/aliases mapped to a shared canonical form, so e.g. a
# resume saying "JS" matches a JD asking for "JavaScript" as an exact hit
# instead of relying on embedding similarity for well-known equivalents.
SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "oop": "object oriented programming",
    "postgres": "postgresql",
    "db": "database",
    "dbs": "databases",
    "ui": "user interface",
    "ux": "user experience",
    "qa": "quality assurance",
    "devops": "development operations",
    "ci/cd": "continuous integration and continuous deployment",
    "cicd": "continuous integration and continuous deployment",
}


def canonicalize(phrase):
    return SKILL_ALIASES.get(phrase, phrase)


def clean_phrase(phrase):
    phrase = re.sub(r"^(a|an|the)\s+", "", phrase.strip())
    for prefix in QUALIFIER_PREFIXES:
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):]
    return phrase.strip()


def is_generic_chunk(chunk):
    """Filter out noun chunks that are filler rather than meaningful skill terms."""
    if chunk.root.pos_ == "PRON":
        return True
    if all(token.is_stop for token in chunk):
        return True
    return False


def extract_candidate_phrases(span):
    """Pull skill-like phrases out of a sentence/doc span, splitting conjunctions."""
    phrases = []

    for chunk in span.noun_chunks:
        if is_generic_chunk(chunk):
            continue
        for part in CONJUNCT_SPLIT.split(chunk.text):
            phrase = clean_phrase(part).lower()
            if phrase in GENERIC_TERMS:
                continue
            # length filter exists to drop junk residue (stray articles,
            # single letters), but must not drop legitimate short
            # abbreviations the alias table depends on (js, ui, ux, ml...).
            if len(phrase) > 2 or phrase in SKILL_ALIASES:
                phrases.append(phrase)

    return phrases


def extract_skills(text):
    # Parsed on the original (not lowercased) text -- spaCy's parser relies
    # on capitalization cues for both sentence boundaries and noun-chunk
    # accuracy. Phrases are lowercased individually as they're extracted.
    doc = nlp(text)
    return list(set(extract_candidate_phrases(doc)))


def extract_weighted_skills(jd_text):
    """
    Extract skill phrases from the JD and weight each one by whether it's
    mentioned in the same sentence as an importance signal (e.g. "required",
    "must have"), rather than just anywhere in the whole document.
    """
    doc = nlp(jd_text)
    weighted = {}

    for sent in doc.sents:
        weight = 3 if any(pattern in sent.text.lower() for pattern in IMPORTANT_PATTERNS) else 1

        for phrase in extract_candidate_phrases(sent):
            # keep the strongest weight seen if a skill is mentioned more than once
            weighted[phrase] = max(weighted.get(phrase, 0), weight)

    return weighted


def semantic_match(resume_skills, jd_weighted, threshold=0.4):
    """
    Match JD skill phrases against resume skill phrases using sentence
    embeddings + cosine similarity, so e.g. "backend development" and
    "building backend services" can match even without exact word overlap.

    threshold=0.4 is an empirical calibration for all-MiniLM-L6-v2 on short
    technical phrases (1-3 words): on a manual sample, true related pairs
    like "relational databases"/"postgresql" (0.45) and "machine learning"/
    "tensorflow" (0.40) scored above it, while unrelated pairs like "docker"/
    "kubernetes" (0.32, related tools but not the same skill) and "sql"/
    "javascript" (0.18) stayed below it. Short-phrase similarity is noisier
    than full-sentence similarity, so don't expect a clean separation --
    this is a reasonable middle ground, not a precise decision boundary.
    """
    if not resume_skills or not jd_weighted:
        return [], list(jd_weighted.keys())

    # Exact-match pass first: canonicalized aliases (e.g. "js" == "javascript")
    # are a guaranteed match and shouldn't be left to embedding-similarity
    # noise, which is unreliable on short technical phrases (see threshold
    # note below).
    resume_canon = {canonicalize(s) for s in resume_skills}

    matched = []
    remaining_jd = []

    for jd_skill in jd_weighted.keys():
        if canonicalize(jd_skill) in resume_canon:
            matched.append(jd_skill)
        else:
            remaining_jd.append(jd_skill)

    missing = []

    if remaining_jd:
        model = get_embedding_model()
        resume_emb = model.encode(resume_skills, convert_to_tensor=True)
        jd_emb = model.encode(remaining_jd, convert_to_tensor=True)

        scores = util.cos_sim(jd_emb, resume_emb)

        for i, jd_skill in enumerate(remaining_jd):
            best_score = float(scores[i].max())
            if best_score >= threshold:
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
    category_scores = {}

    for category, keywords in SKILL_CATEGORIES.items():
        count = sum(1 for s in matched if any(k in s.lower() for k in keywords))
        category_scores[category] = count

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
