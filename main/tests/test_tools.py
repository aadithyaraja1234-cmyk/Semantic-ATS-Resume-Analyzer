import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import (
    calculate_weighted_score,
    calculate_confidence,
    extract_years_experience,
    detect_leadership,
    detect_impact_metrics,
    categorize_skills,
    canonicalize,
    semantic_match,
    extract_skills,
    extract_weighted_skills,
    find_taxonomy_skills,
    compare_skills,
)
from agent_brain import get_recommendation, get_risk


def test_calculate_weighted_score_full_match():
    jd_weighted = {"python": 3, "amazon web services": 1}
    matched = ["python", "amazon web services"]
    assert calculate_weighted_score(matched, jd_weighted) == 100.0


def test_calculate_weighted_score_partial_match():
    jd_weighted = {"python": 3, "amazon web services": 1}
    matched = ["python"]
    assert calculate_weighted_score(matched, jd_weighted) == 75.0


def test_calculate_weighted_score_no_requirements_is_zero():
    assert calculate_weighted_score([], {}) == 0


def test_extract_years_experience_picks_the_max():
    text = "3 years of experience in support, 6 years experience in backend engineering"
    assert extract_years_experience(text) == 6


def test_extract_years_experience_defaults_to_zero():
    assert extract_years_experience("Experienced backend engineer") == 0


def test_detect_leadership_true():
    assert detect_leadership("Led a team of 5 engineers") is True


def test_detect_leadership_false():
    assert detect_leadership("Wrote unit tests and fixed bugs") is False


def test_detect_impact_metrics_percentage():
    assert detect_impact_metrics("Improved performance by 40%") is True


def test_detect_impact_metrics_dollar_amount():
    assert detect_impact_metrics("Saved the company $50k annually") is True


def test_detect_impact_metrics_absent():
    assert detect_impact_metrics("Worked on the backend team") is False


def test_categorize_skills_counts_by_category():
    matched = ["python", "docker", "amazon web services"]
    result = categorize_skills(matched)
    assert result["Programming Languages"] == 1
    assert result["DevOps & CI/CD"] == 1
    assert result["Cloud & Infrastructure"] == 1


def test_categorize_skills_zero_for_uninvolved_categories():
    result = categorize_skills(["python"])
    assert result["Security"] == 0
    assert result["Design & UX"] == 0


def test_get_recommendation_bands():
    assert get_recommendation(90) == "Strong Fit"
    assert get_recommendation(70) == "Good Fit"
    assert get_recommendation(50) == "Moderate Fit"
    assert get_recommendation(20) == "Low Fit"


def test_get_risk_bands():
    assert get_risk(85) == "Low Risk"
    assert get_risk(65) == "Medium Risk"
    assert get_risk(30) == "High Risk"


def test_compare_skills_returns_expected_shape():
    """
    Regression test for the bug that shipped: streamlit_app.py read
    result["years"]/["leadership"]/["impact"], but compare_skills() never
    produced them, so every analysis crashed with a KeyError.
    """
    resume = "Experienced Python developer with AWS and Docker skills. 5 years of experience."
    jd = "We need a candidate with strong experience in Python and AWS. Docker is a plus."

    result = compare_skills(resume, jd)

    assert set(result.keys()) == {
        "matched", "missing", "score", "confidence", "categories",
        "years", "leadership", "impact"
    }
    assert 0 <= result["score"] <= 100
    assert isinstance(result["matched"], list)
    assert isinstance(result["missing"], list)
    assert result["years"] == 5


def test_find_taxonomy_skills_ignores_non_skill_text():
    """
    Regression test for the bug that prompted this rewrite: noun-chunk
    extraction was picking up things like "35%" and "senior backend
    engineer" as if they were skills. Taxonomy matching only ever returns
    known skill names, so junk like this can't appear at all.
    """
    text = "Senior Backend Engineer, improved latency by 35%, strong background, 6 years of experience."
    skills = find_taxonomy_skills(text)
    assert "35%" not in skills
    assert "improved latency" not in skills
    assert "senior backend engineer" not in skills
    assert "strong background" not in skills


def test_find_taxonomy_skills_word_boundary_safe():
    """"java" must not match inside "javascript", and vice versa isn't an issue
    since they're both real distinct taxonomy entries."""
    skills = find_taxonomy_skills("Experienced in JavaScript development.")
    assert "javascript" in skills
    assert "java" not in skills


def test_extract_skills_resolves_short_alias_abbreviations():
    skills = extract_skills("Experienced in Python and JS, with a focus on UX.")
    assert "javascript" in skills
    assert "user experience design" in skills


def test_canonicalize_maps_known_aliases():
    assert canonicalize("js") == "javascript"
    assert canonicalize("aws") == "amazon web services"
    assert canonicalize("python") == "python"  # no alias -> unchanged


def test_alias_abbreviations_match_across_resume_and_jd():
    """
    A resume that says "JS" should match a JD that says "JavaScript" -- both
    resolve to the same canonical name, so this is an exact match, not an
    embedding-similarity guess.
    """
    resume_skills = extract_skills("Experienced in Python and JS development.")
    jd_weighted = extract_weighted_skills("Looking for strong JavaScript experience.")

    matched, missing = semantic_match(resume_skills, jd_weighted)

    assert "javascript" in matched
    assert missing == []


def test_extract_weighted_skills_weights_by_sentence_importance():
    jd = "Must have Python experience. Docker knowledge is a plus."
    weighted = extract_weighted_skills(jd)
    assert weighted["python"] == 3
    assert weighted["docker"] == 1


def test_calculate_confidence_scales_with_signal_size():
    # Same score, but far more extracted skill terms -> higher confidence.
    low_signal = calculate_confidence(score=90, jd_weighted={"python": 1}, resume_skills=["python"])
    high_signal = calculate_confidence(
        score=90,
        jd_weighted={f"skill{i}": 1 for i in range(6)},
        resume_skills=[f"skill{i}" for i in range(6)]
    )
    assert low_signal < high_signal
    assert high_signal == 90.0  # 12+ combined terms -> full confidence


def test_calculate_confidence_zero_signal_is_zero():
    assert calculate_confidence(score=100, jd_weighted={}, resume_skills=[]) == 0


class TestSemanticMatchCalibration:
    """
    Locks in the empirical threshold=0.4 calibration for all-MiniLM-L6-v2 on
    short skill-name phrases (see semantic_match()'s docstring). If the
    model or threshold changes, these should be re-validated, not just
    adjusted to pass -- they're the actual evidence behind the "0.4" choice.
    """

    def test_related_pairs_match(self):
        matched, _ = semantic_match(["postgresql"], {"relational databases": 1})
        assert matched == ["relational databases"]

        matched, _ = semantic_match(["tensorflow"], {"machine learning": 1})
        assert matched == ["machine learning"]

    def test_unrelated_pairs_do_not_match(self):
        _, missing = semantic_match(["javascript"], {"sql": 1})
        assert missing == ["sql"]

        _, missing = semantic_match(["docker"], {"kubernetes": 1})
        assert missing == ["kubernetes"]
