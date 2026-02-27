from tools import compare_skills
from llm_layer import generate_agent_response


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
    confidence = round(data["score"] * 0.9, 2)

    prompt = f"""
Evaluate professionally.

Matched Skills:
{data["matched"]}

Missing Skills:
{data["missing"]}

Match Score: {data["score"]}%

Provide concise professional evaluation.
"""

    llm_response = generate_agent_response(prompt)

    data.update({
        "recommendation": recommendation,
        "risk": risk,
        "confidence": confidence,
        "analysis": llm_response
    })

    return data
