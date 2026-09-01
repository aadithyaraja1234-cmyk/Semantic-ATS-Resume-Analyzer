# 📄 Semantic ATS Resume Analyzer

An AI-powered tool that scores how well a resume matches a job description —
using **sentence embeddings** for semantic skill matching (not just keyword
overlap), weighted scoring based on how critical each requirement is in the
JD, and an LLM-generated recruiter-style evaluation.

## Features

- **Upload or paste** a resume (PDF, DOCX, or plain text) and a job description
- **Semantic skill matching** — matches skills by meaning using sentence
  embeddings (`all-MiniLM-L6-v2`), so e.g. *"building backend services"* can
  match *"backend development"* even without exact word overlap; common
  abbreviations (*"JS"* ↔ *"JavaScript"*, *"k8s"* ↔ *"Kubernetes"*) match
  exactly via an alias table rather than relying on embedding similarity
- **Confidence that reflects signal, not just score** — a high match score
  built on only 1-2 extracted requirements is flagged as lower-confidence
  than the same score built on a dozen
- **Weighted match score** — requirements mentioned alongside language like
  *"required"* or *"must have"* count more than nice-to-haves
- **Skill categorization** — buckets matched skills into Cloud, DevOps,
  Backend, ML, and Database
- **Resume intelligence** — years of experience, leadership signals, and
  quantified-impact detection, extracted directly from the resume text
- **AI evaluation** — an LLM (via [LiteLLM](https://github.com/BerriAI/litellm),
  Groq by default) writes a structured strength/gap/recommendation summary
- **Fails gracefully** — the score, matched/missing skills, and resume
  intelligence all work with zero API keys configured; only the AI
  evaluation section degrades if no LLM key is set

## How it works

```
Resume text ─┐                                   ┌─→ Matched / Missing skills
             ├─→ spaCy noun-chunk extraction ─┐   │
JD text ─────┘                                ├─→ Sentence-embedding      ├─→ Weighted match score
                                               │   cosine similarity      │
JD sentences ─→ "required"/"must have" scan ───┘   (all-MiniLM-L6-v2)     ├─→ Recommendation / Risk
                                                                          │
Resume text ─→ regex: years / leadership / impact signals ───────────────┴─→ Resume intelligence
                                                                          
Matched + missing skills + score ─→ LLM prompt ─→ AI evaluation (LiteLLM)
```

1. **Skill extraction** — spaCy pulls noun phrases out of both texts, filtering
   generic filler ("years of experience", "our team") so the lists stay
   focused on real skill terms.
2. **Importance weighting** — each JD skill is weighted 3x if it appears in a
   sentence containing an importance signal ("required", "must have",
   "mandatory", "strong experience"), otherwise 1x.
3. **Semantic matching** — resume and JD skill phrases are first compared via
   an alias table (`js` ↔ `javascript`, `k8s` ↔ `kubernetes`, etc.) for exact
   matches, then any that don't match are embedded with `sentence-transformers`
   and compared via cosine similarity, so paraphrases and synonyms still
   match without needing to be identical strings.
4. **Scoring** — the match score is the weighted sum of matched JD skills
   over the total weight of all JD skills; confidence then scales that score
   down when it's based on very few extracted skill terms.
5. **AI evaluation** — the matched/missing skills and score are handed to an
   LLM (via LiteLLM) to generate a structured, recruiter-style writeup.

## Tech Stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| NLP / skill extraction | [spaCy](https://spacy.io/) (`en_core_web_sm`) |
| Semantic matching | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| LLM evaluation | [LiteLLM](https://github.com/BerriAI/litellm) (Groq by default; swappable) |
| File parsing | [pypdf](https://pypdf.readthedocs.io/), [python-docx](https://python-docx.readthedocs.io/) |
| Tests | [pytest](https://docs.pytest.org/) |

## Setup

**Requires Python 3.10 or 3.11.** `pip install` will fail on Python 3.13+ --
`blis` (spaCy's linear-algebra backend) has no prebuilt wheel for 3.13 yet,
so pip tries to compile it from source and fails without a C build
toolchain. This is an upstream packaging gap (confirmed against the latest
spaCy release, not just the pinned one here), not something fixable from
this repo alone -- use 3.10 or 3.11 until that changes.

If `python --version` shows something newer, install 3.11 alongside it
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
├── tools.py             # Skill extraction, semantic matching, scoring
├── llm_layer.py          # LiteLLM wrapper with provider-aware error handling
├── file_parser.py         # PDF/DOCX/TXT text extraction
├── main.py                 # CLI entry point
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt      # + pytest
└── tests/
    ├── test_tools.py          # Unit + regression tests, incl. the semantic-
    │                            match threshold calibration (see below)
    └── test_file_parser.py     # PDF/DOCX/TXT extraction tests
```

## Limitations & Roadmap

- **Skill extraction is noun-phrase based (spaCy) plus a small alias table**,
  not a curated skills taxonomy or trained NER model — it can still surface
  phrases that aren't real skills on unusual inputs, and the alias table only
  covers common abbreviations, not every possible synonym.
- **The 0.4 similarity threshold is an empirical calibration on a small
  manual sample** (documented and regression-tested in `test_tools.py`), not
  a threshold tuned against a labeled precision/recall dataset. Short
  technical phrases (1-3 words) don't embed as cleanly as full sentences, so
  don't expect a razor-sharp match/no-match boundary.
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
