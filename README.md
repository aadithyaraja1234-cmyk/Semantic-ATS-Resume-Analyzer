# 📄 Semantic ATS Resume Analyzer

An AI-powered tool that scores how well a resume matches a job description —
using **sentence embeddings** for semantic skill matching (not just keyword
overlap), weighted scoring based on how critical each requirement is in the
JD, and an LLM-generated recruiter-style evaluation.

## Features

- **Upload or paste** a resume (PDF, DOCX, or plain text) and a job description
- **Curated skill taxonomy** — ~130 real skills across 15 categories
  (Programming Languages, Cloud & Infrastructure, DevOps & CI/CD, Data
  Science & ML, Security, Soft Skills, and more) are matched by exact name
  or known alias (*"JS"* ↔ *"JavaScript"*, *"k8s"* ↔ *"Kubernetes"*), so
  results are always real skill names — never stray fragments like a
  percentage or a job title
- **Semantic fallback matching** — anything not caught by the taxonomy/alias
  pass falls back to sentence embeddings (`all-MiniLM-L6-v2`), so e.g.
  *"PostgreSQL"* still credits a JD asking for *"relational databases"*
  even without an exact name match
- **Confidence that reflects signal, not just score** — a high match score
  built on only 1-2 extracted requirements is flagged as lower-confidence
  than the same score built on a dozen
- **Weighted match score** — requirements mentioned alongside language like
  *"required"* or *"must have"* count more than nice-to-haves
- **Skill categorization** — matched skills are grouped into their taxonomy
  category and charted, instead of a flat, mostly-empty keyword count
- **Resume intelligence** — years of experience, leadership signals, and
  quantified-impact detection, extracted directly from the resume text
- **AI evaluation** — an LLM (via [LiteLLM](https://github.com/BerriAI/litellm),
  Groq by default) writes a structured strength/gap/recommendation summary
- **Fails gracefully** — the score, matched/missing skills, and resume
  intelligence all work with zero API keys configured; only the AI
  evaluation section degrades if no LLM key is set

## How it works

```
Resume text ─┐                                          ┌─→ Matched / Missing skills
             ├─→ Curated skill taxonomy (exact + alias) ─┤
JD text ─────┘   ~130 skills across 15 categories        ├─→ Weighted match score
                                                          │
JD sentences ─→ "required"/"must have" scan ─────────────┤
                                                          ├─→ Recommendation / Risk / Confidence
Unmatched JD skills ─→ sentence-embedding fallback ───────┤   (all-MiniLM-L6-v2)
                                                          │
Resume text ─→ regex: years / leadership / impact signals ┴─→ Resume intelligence

Matched + missing skills + score ─→ LLM prompt ─→ AI evaluation (LiteLLM)
```

1. **Skill extraction** — both texts are checked against a curated taxonomy
   (`skills_taxonomy.py`) of canonical skill names and their common aliases,
   using word-boundary-safe matching (so "java" never matches inside
   "javascript"). Only real skill names can ever come out of this step.
2. **Importance weighting** — the JD is split into sentences; each skill
   found is weighted 3x if its sentence contains an importance signal
   ("required", "must have", "mandatory", "strong experience"), otherwise 1x.
3. **Matching** — resume and JD skills are already canonical taxonomy names
   at this point, so matching is a direct exact comparison first. Anything
   left over falls back to embedding similarity between the clean skill
   names (not noisy free text), catching close synonyms the taxonomy/alias
   table doesn't explicitly list.
4. **Scoring** — the match score is the weighted sum of matched JD skills
   over the total weight of all JD skills; confidence then scales that score
   down when it's based on very few extracted skill terms.
5. **AI evaluation** — the matched/missing skills and score are handed to an
   LLM (via LiteLLM) to generate a structured, recruiter-style writeup.

## Tech Stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Skill extraction | Curated taxonomy (`skills_taxonomy.py`) + regex matching |
| Semantic fallback matching | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| LLM evaluation | [LiteLLM](https://github.com/BerriAI/litellm) (Groq by default; swappable) |
| File parsing | [pypdf](https://pypdf.readthedocs.io/), [python-docx](https://python-docx.readthedocs.io/) |
| Tests | [pytest](https://docs.pytest.org/) |

## Setup

**Requires Python 3.11+.** `litellm` imports `typing.NotRequired`, which only
exists in the standard library from Python 3.11 onward (PEP 655) -- so 3.10
and earlier fail at import time. (An earlier version of this app also
depended on spaCy, which added its own Python-version constraints on top of
this one -- that dependency has since been removed in favor of a curated
skill taxonomy, so this is now the only version constraint.)

If `python --version` shows something older, install 3.11 alongside it
rather than replacing your default:

```bash
# Windows (py launcher, installed automatically with Python from python.org)
py -3.11 -m venv .venv
.venv\Scripts\activate

# macOS/Linux (pyenv)
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
```

Then, from the activated environment:

```bash
git clone https://github.com/aadithyaraja1234-cmyk/Semantic-ATS-Resume-Analyzer.git
cd Semantic-ATS-Resume-Analyzer
pip install -r main/requirements.txt
```

Copy `.env.example` to `.env` and set your LLM API key (a free key works at
[console.groq.com/keys](https://console.groq.com/keys)):

```bash
cp .env.example .env
```

The app runs fine without a key configured — the score, skill matching, and
resume intelligence sections work regardless; only the "AI Evaluation"
section will show a message telling you to set one.

## Usage

```bash
streamlit run main/streamlit_app.py
```

Or from the command line:

```bash
python main/main.py
```

## Running Tests

```bash
pip install -r main/requirements-dev.txt
pytest main/tests -v
```

Tests run automatically on every push via [GitHub Actions](.github/workflows/tests.yml).

## Project Structure

```
main/
├── streamlit_app.py   # UI: upload/paste resume + JD, display results
├── agent_brain.py      # Orchestrates scoring + LLM evaluation
├── tools.py             # Skill extraction, matching, scoring
├── skills_taxonomy.py    # Curated skill list: canonical name -> category + aliases
├── llm_layer.py           # LiteLLM wrapper with provider-aware error handling
├── file_parser.py          # PDF/DOCX/TXT text extraction
├── main.py                  # CLI entry point
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt       # + pytest
└── tests/
    ├── test_tools.py           # Unit + regression tests, incl. the semantic-
    │                             match threshold calibration (see below)
    └── test_file_parser.py      # PDF/DOCX/TXT extraction tests
```

## Limitations & Roadmap

- **The taxonomy covers ~130 common skills across 15 categories** — broad,
  but not exhaustive. A skill entirely outside both the taxonomy and its
  aliases won't be recognized at all, even via the embedding fallback (which
  only compares *already-recognized* skills against each other). Extending
  coverage means adding entries to `skills_taxonomy.py`.
- **The 0.4 similarity threshold is an empirical calibration on a small
  manual sample** (documented and regression-tested in `test_tools.py`), not
  a threshold tuned against a labeled precision/recall dataset.
- A few taxonomy terms are ambiguous out of context (e.g. "Excel" as a skill
  vs. the verb) — a minor source of false positives, same tradeoff any
  keyword-based ATS tool makes.
- Scanned/image-only PDFs have no extractable text (no OCR yet).
- Only one resume vs. one JD at a time — no batch/bulk comparison.
- Possible next steps: a proper labeled evaluation set for the matching
  threshold, a larger curated skill taxonomy, OCR fallback for scanned PDFs,
  batch scoring against multiple JDs, and a downloadable PDF report.

## Privacy

Nothing uploaded or pasted is stored — it exists only in memory for the
session and is discarded when the tab closes. The raw resume/JD text never
leaves the app; only the extracted skill list and match score are sent to
the AI evaluation step (via whichever LLM provider is configured), and only
if that step runs. No accounts, cookies, or analytics are used.

## Contact

Questions, bugs, or ideas: open an [issue](https://github.com/aadithyaraja1234-cmyk/Semantic-ATS-Resume-Analyzer/issues)
or reach out via [GitHub](https://github.com/aadithyaraja1234-cmyk).

## License

[MIT](LICENSE)
