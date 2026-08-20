from tools import compare_skills
from llm_layer import generate_agent_response, LLMConfigError


def get_recommendation(score):
    if score >= 85:
        return "Strong Fit"
    elif score >= 65:
        return "Good Fit"
    elif score >= 45:
        return "Moderate Fit"
    else:
        return "Low Fit"


def get_risk(score):
    if score >= 80:
        return "Low Risk"
    elif score >= 60:
        return "Medium Risk"
    else:
        return "High Risk"


def resume_agent(resume_text, job_description):

    data = compare_skills(resume_text, job_description)

    recommendation = get_recommendation(data["score"])
    risk = get_risk(data["score"])
    # data["confidence"] is already computed by compare_skills(), based on
    # how much skill signal the score is built on -- not recomputed here.

    prompt = f"""
Evaluate professionally.

Matched Skills:
{data["matched"]}

Missing Skills:
{data["missing"]}

Match Score: {data["score"]}%

Provide concise professional evaluation.
"""

    # The skill-matching/scoring above never depends on the LLM, so a missing
    # API key or provider outage shouldn't take down the whole analysis --
    # only the AI evaluation section degrades.
    try:
        analysis = generate_agent_response(prompt)
        analysis_error = False
    except LLMConfigError as e:
        analysis = str(e)
        analysis_error = True

    data.update({
        "recommendation": recommendation,
        "risk": risk,
        "analysis": analysis,
        "analysis_error": analysis_error
    })

    return data
