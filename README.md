# FOA Ingestion & Rule-Based NLP Tagging (HumanAI ISSR4 Playground)

This repository contains a minimal script that ingests a single FOA URL, extracts fields into a fixed schema, applies deterministic rule-based tags using lightweight NLP, and exports both JSON and CSV.

The current implementation targets **Grants.gov** FOA detail URLs (e.g. `https://www.grants.gov/search-results-detail/361087`). The URL router is written so that an NSF extractor can be added later without changing the CLI.

---

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal:

- `requests` — for calling the public Grants.gov `fetchOpportunity` API.

---

## Usage

Run the program as specified in the prompt:

```bash
python main.py --url "https://www.grants.gov/search-results-detail/361087" --out_dir ./out
```

On success, `out/` will contain:

- `foa.json` — single-record JSON with all extracted fields and nested tags
- `foa.csv` — one-row CSV with the same fields, plus flattened tag columns

CLI arguments:

- `--url` (required): FOA detail URL from Grants.gov.
- `--out_dir` (required): Output directory path (created if it does not exist).

---

## Data schema

The core FOA schema is defined by the `FOARecord` dataclass in `main.py`:

- `foa_id`
- `title`
- `agency`
- `posting_date`
- `close_date`
- `award_ceiling`
- `award_floor`
- `description`
- `eligibility`
- `source` (e.g. `"grants.gov"`)
- `source_url` (the original URL)
- `retrieved_at` (UTC ISO 8601 timestamp)
- `tags` (nested dict, see below)

In the CSV export, tags are flattened into columns named:

- `tags_<category>` (e.g. `tags_methods`, `tags_sponsor_themes`)

Each such column contains a `;`-separated list of tag names.

---

## Extraction logic (Grants.gov)

`main.py` implements a `GrantsGovExtractor` that:

- Parses the numeric `opportunityId` from the Grants.gov FOA detail URL.
- Calls the public `https://api.grants.gov/v1/api/fetchOpportunity` endpoint.
- Normalizes the response JSON into the `FOARecord` schema:
  - `foa_id`, `title`, `agency`
  - `posting_date`, `close_date`
  - `award_ceiling`, `award_floor`
  - `description`, `eligibility` (from `applicantTypes`)
  - `source`, `source_url`, `retrieved_at`

Non-Grants.gov URLs will raise a clear error; NSF support is intentionally out of scope for this script and can be implemented separately (e.g., in a dedicated notebook).

---

## Deterministic, rule-based NLP tagging (no ML models)

Tagging is handled by `RuleBasedTagger` in `main.py`. It is **strictly deterministic and rule-based**:

- All behavior is encoded in a small ontology of **hand-written keyword rules with weights**.
- There are **no machine-learning models, embeddings, or external APIs** involved in tagging.
- Given the same FOA text and ontology, the tagger will always return **the exact same tags and scores**.

At the same time, it uses light NLP-style processing to make the rules more robust:

Key details:

- Inputs: concatenation of `title`, `description`, and `eligibility` (when present).
- Normalization:
  - Lowercase
  - Collapse whitespace
- Matching (still fully rule-based):
  - Word-boundary-aware regexes for patterns, so `"health"` does not match inside `"wealth"`.
  - Optional plural `"s"` for single-word patterns (e.g. `"veteran"` vs `"veterans"`).
  - Multi-word phrases (e.g. `"natural language"`, `"public health"`) matched as anchored phrases.
- Ontology:
  - `sponsor_themes` (e.g. **Public Health**, **Mental Health**)
  - `research_domains` (e.g. **Clinical Research**, **Data Science**)
  - `methods` (e.g. **NLP**, **Machine Learning**)
  - `populations` (e.g. **Youth**, **Veterans**)
  - Each tag is defined by multiple `KeywordRule` objects with weights.
- Scoring and dominant-category selection:
  - For each tag, compute `score = matched_weight / total_weight` and clip to \[0, 1\].
  - Only tags above a per-category threshold (e.g. 0.35) are kept.
  - Tags within a category are sorted by score and truncated to top-k (default 3).
  - Then, categories are compared by their **max tag score**, and only the
    **dominant semantic category** (or categories within 80% of the best score)
    are retained in the final `tags` dict. This means that for a FOA whose text
    is mostly about a research domain, you will typically see only
    `research_domains` populated, not every possible category.

The nested `tags` structure in JSON looks like:

```json
{
  "methods": [
    {"name": "NLP", "score": 0.9},
    {"name": "Machine Learning", "score": 0.7}
  ],
  "populations": [
    {"name": "Youth", "score": 0.6}
  ]
}
```

In the CSV, these become columns such as:

- `tags_methods = "NLP;Machine Learning"`
- `tags_populations = "Youth"`

---

## Extending / improving the NLP

To tweak or extend the tagging:

- Add new categories or tags to `self.ontology` in `RuleBasedTagger`.
- For each tag:
  - Add more `KeywordRule` entries to cover synonyms and related phrases.
  - Adjust `weight` to emphasize especially diagnostic phrases.
- Adjust per-category `threshold` if you want more or fewer tags returned.

You can also experiment in the notebook `humanai_issr4_grantsgov.ipynb` and then copy the final ontology back into `main.py`.

