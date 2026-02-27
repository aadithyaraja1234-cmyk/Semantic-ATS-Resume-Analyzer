import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "groq/llama-3.1-8b-instant")

SYSTEM_PROMPT = """
You are a senior technical recruiter evaluating resumes professionally.

Rules:
- Be concise and analytical.
- Do NOT repeat the resume content.
- Do NOT use generic phrases.
- Focus only on skill alignment and job readiness.
- Avoid conversational tone.
- Provide structured sections with headers.

Output format:

1. Strength Assessment (2-3 lines)
2. Skill Gaps (if any)
3. Targeted Improvement Recommendations (bullet points)
4. Final Score (X/10 with 1-line justification)

Be precise and professional.
"""

def generate_agent_response(prompt):
    response = completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )

    return response["choices"][0]["message"]["content"]