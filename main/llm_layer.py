import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "groq/openai/gpt-oss-20b")

# Maps a LiteLLM provider prefix (the part before "/" in MODEL_NAME) to the
# environment variable that provider expects an API key in. Used to give a
# clear, actionable error instead of a raw exception when a key is missing.
PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

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


class LLMConfigError(Exception):
    """Raised when the AI evaluation step can't run (missing key, provider error, etc.)."""
    pass


def _required_env_var():
    provider = MODEL_NAME.split("/")[0] if "/" in MODEL_NAME else None
    return PROVIDER_ENV_VARS.get(provider)


def generate_agent_response(prompt):
    env_var = _required_env_var()
    if env_var and not os.getenv(env_var):
        raise LLMConfigError(
            f"Missing {env_var}. Copy .env.example to .env and set it to enable AI evaluation."
        )

    try:
        response = completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            # Some models (e.g. openai/gpt-oss-*) spend part of this budget on
            # hidden reasoning tokens before the visible answer -- 1200 gives
            # headroom beyond that so long skill lists don't get cut off.
            max_tokens=1200
        )
        content = response["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise LLMConfigError(
                "The AI model returned an empty response (likely spent its "
                "whole token budget on internal reasoning). Try again."
            )
        return content
    except LLMConfigError:
        raise
    except Exception as e:
        raise LLMConfigError(f"AI evaluation failed: {e}") from e
