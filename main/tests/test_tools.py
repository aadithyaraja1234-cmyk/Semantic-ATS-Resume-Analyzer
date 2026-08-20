import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import (
    clean_phrase,
    calculate_weighted_score,
    calculate_confidence,
    extract_years_experience,
    detect_leadership,
    detect_impact_metrics,
    categorize_skills,
    canonicalize,
    semantic_match,
    extract_skills,
    compare_skills,
)
from agent_brain import get_recommendation, get_risk


def test_clean_phrase_strips_leading_articles():
    assert clean_phrase("the python") == "python"
    assert clean_phrase("an api") == "api"
    assert clean_phrase("a database") == "database"
    assert clean_phrase("kubernetes") == "kubernetes"


def test_calculate_weighted_score_full_match():
    jd_weighted = {"python": 3, "aws": 1}
    matched = ["python", "aws"]
    assert calculate_weighted_score(matched, jd_weighted) == 100.0


def test_calculate_weighted_score_partial_match():
    jd_weighted = {"python": 3, "aws": 1}
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
    matched = ["python", "docker", "aws"]
    result = categorize_skills(matched)
    assert result["Backend"] == 1
    assert result["DevOps"] == 1
    assert result["Cloud"] == 1


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


def test_extract_skills_keeps_short_alias_abbreviations():
    """
    Regression test: the length filter meant to drop junk residue (stray
    articles, single letters) was also silently dropping legitimate 2-char
    alias abbreviations like "js"/"ui"/"ux" before they ever reached
    SKILL_ALIASES, making the alias table dead code for exactly the terms
    it exists to handle.
    """
    skills = extract_skills("Experienced in Python and JS, with a focus on UX.")
    assert "js" in skills
    assert "ux" in skills


def test_canonicalize_maps_known_aliases():
    assert canonicalize("js") == "javascript"
    assert canonicalize("aws") == "amazon web services"
    assert canonicalize("python") == "python"  # no alias -> unchanged


def test_semantic_match_exact_alias_bypasses_embeddings():
    """
    Alias matches (e.g. "js" resume skill vs "javascript" JD requirement)
    should match via exact canonical-form comparison, not embedding
    similarity -- this must hold even if the embedding model is swapped out.
    """
    matched, missing = semantic_match(["js"], {"javascript": 3})
    assert matched == ["javascript"]
    assert missing == []


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
    short technical phrases (see semantic_match()'s docstring). If the model
    or threshold changes, these should be re-validated, not just adjusted to
    pass -- they're the actual evidence behind the "0.4" choice.
    """

    def test_related_pairs_match(self):
        matched, _ = semantic_match(["postgresql"], {"relational databases": 1})
        assert matched == ["relational databases"]

        matched, _ = semantic_match(["tensorflow"], {"machine learning": 1})
        assert matched == ["machine learning"]

    def test_unrelated_pairs_do_not_match(self):
        _, missing = semantic_match(["javascript"], {"sql": 1})
        assert missing == ["sql"]

        _, missing = semantic_match(["photoshop"], {"aws": 1})
        assert missing == ["aws"]
