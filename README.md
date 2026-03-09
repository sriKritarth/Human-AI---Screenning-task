# FOA Ingestion & Rule-Based NLP Tagging (HumanAI ISSR4 Playground)

This repository contains a small playground for FOA ingestion and tagging:

- **CLI pipeline (`main.py`)**: ingests a single **Grants.gov** FOA URL, normalizes it into a fixed schema, applies **deterministic rule-based tags** using a YAML ontology, and exports JSON + CSV.
- **NSF notebook (`humanAI_issr4_nsf.ipynb`)**: explores FOAs from **NSF (nsf.gov)** using an HTML scraper, the *same deterministic tagger*, and an additional **semantic, embedding-based tagging pass** built on `sentence-transformers` to compare / augment tags.

---

## Requirements

- Python 3.9+
- Install core dependencies (CLI + ontology + HTML parsing):

```bash
pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal and supports the deterministic pipeline:

- `requests` — for calling the public Grants.gov `fetchOpportunity` API.
- `PyYAML` — for loading the tagging ontology from `ontologies.yaml`.
- `beautifulsoup4` — used by the NSF notebook to parse HTML from `nsf.gov`.
- `ipykernel` — to run the notebook in this environment.

For the **semantic tagging part of the NSF notebook**, you will also need (installed inside your notebook kernel or environment):

```bash
pip install sentence-transformers
```

If you have a Hugging Face account, you can optionally set `HF_TOKEN` in your environment to get faster, authenticated model downloads and higher rate limits.

---

## Usage

Run the deterministic Grants.gov pipeline as specified in the prompt:

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

The core FOA schema for the CLI pipeline is defined by the `FOARecord` dataclass in `main.py`:

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

## Extraction logic (Grants.gov CLI)

`main.py` implements a `GrantsGovExtractor` that:

- Parses the numeric `opportunityId` from the Grants.gov FOA detail URL.
- Calls the public `https://api.grants.gov/v1/api/fetchOpportunity` endpoint.
- Normalizes the response JSON into the `FOARecord` schema:
  - `foa_id`, `title`, `agency`
  - `posting_date`, `close_date`
  - `award_ceiling`, `award_floor`
  - `description`, `eligibility` (from `applicantTypes`)
  - `source`, `source_url`, `retrieved_at`

Non-Grants.gov URLs will raise a clear error; NSF support is **not** wired into the CLI. Instead, NSF ingestion and tagging lives in a dedicated exploratory notebook (`humanAI_issr4_nsf.ipynb`).

---

## Deterministic, rule-based NLP tagging (no ML models)

Deterministic tagging for **both Grants.gov and NSF FOAs** is handled by `RuleBasedTagger` (defined in `main.py` and imported into the NSF notebook). It is **strictly deterministic and rule-based**:

- All behavior is encoded in an ontology described in **`ontologies.yaml`** (hand-written keyword rules with weights).
- There are **no machine-learning models, embeddings, or external APIs** involved in the deterministic tagging path.
- Given the same FOA text and ontology, the tagger will always return **the exact same tags**.

At the same time, it uses light NLP-style processing to make the rules more robust:

Key details (Grants.gov CLI pipeline):

- Inputs: concatenation of `title`, `description`, `eligibility`, and `agency` (when present) to give the ontology more signal.
- Normalization:
  - Lowercase
  - Collapse whitespace
- Matching (still fully rule-based, derived from `ontologies.yaml`):
  - Word-boundary-aware regexes for patterns, so `"health"` does not match inside `"wealth"`.
  - Optional plural `"s"` for single-word patterns (e.g. `"veteran"` vs `"veterans"`).
  - Multi-word phrases (e.g. `"natural language"`, `"public health"`) matched as anchored phrases.
- Ontology (defined in `ontologies.yaml`):
  - `sponsor_themes` (e.g. **Public Health**, **Mental Health**)
  - `research_domains` (e.g. **Cyberinfrastructure**, **Computational**)
  - `methods` (e.g. **NLP**, **Geospatial**)
  - `populations` (e.g. **Youth**, **Veterans**)
  - `funding_mechanism` (e.g. **Research Grant**, **Cooperative Agreement**, **Contract**)
  - `data_types` (e.g. **Administrative Data**, **Survey / Interview Data**, **Text / Document Data**)
  - `settings` (e.g. **Criminal Justice System**, **Education Settings**)
  - `tech_focus` (e.g. **AI / Machine Learning**, **Cybersecurity / Privacy**, **Standardization / Interoperability**)
  - `sponsor_org` (e.g. **NIJ / DOJ**, **NSF**, **NIH**)
  - Each tag is defined by multiple `keywords` with patterns and weights.
- Scoring and dominant-category selection:
  - For each tag, compute `score = matched_weight / total_weight` and clip to \[0, 1\]. (This logic is implemented via keyword rules loaded from the YAML.)
  - Only tags above a per-category threshold (e.g. 0.35) are kept.
  - Tags within a category are sorted by score and truncated to top-k (default 3).
  - Then, categories are compared by their **max tag score**, and only the
    **dominant semantic category** (or categories within 80% of the best score)
    are retained in the final `tags` dict. This means that for a FOA whose text
    is mostly about a research domain, you will typically see only
    `research_domains` populated, not every possible category.

The nested `tags` structure in JSON (before flattening) looks like:

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

## NSF notebook: HTML extraction + deterministic + semantic tagging

The notebook `humanAI_issr4_nsf.ipynb` explores a different data source (**NSF Dear Colleague Letters on `nsf.gov`**) and a hybrid tagging approach:

- **Extraction (NSF)**:
  - Defines an `NSFFOARecord` dataclass similar to `FOARecord` but tailored for NSF FOAs.
  - Uses `requests` + `BeautifulSoup` to fetch and parse HTML from specific NSF URLs (e.g. Dear Colleague Letters).
  - Extracts title, a short description, and, where available, `ELIGIBILITY` / `DEADLINES` text blocks directly from the page.
- **Deterministic tagging for NSF**:
  - Wraps `RuleBasedTagger` in a helper `tag_nsf_foa(foa: NSFFOARecord)`.
  - Concatenates key NSF fields (`title`, `description`, `eligibility`, `agency`) into a single text string.
  - Calls the **same deterministic ontology** (from `ontologies.yaml`) used by the Grants.gov CLI, returning a record with a `tags` dict in the same shape as the CLI output.
- **Semantic tag suggestions (sentence-transformers)**:
  - Builds a list of label texts from the ontology (e.g. `"tech_focus :: AI / Machine Learning"`).
  - Encodes these labels with a `SentenceTransformer` model (e.g. `all-MiniLM-L6-v2`) to create an embedding index.
  - For each NSF FOA, encodes the FOA text and computes cosine similarity against the label embeddings.
  - Produces a ranked list of **semantic suggestions** of the form `{category, name, semantic_score}`.
- **Hybrid tagging**:
  - A helper `augment_tags_with_semantic` takes:
    - The deterministic tags (from `tag_nsf_foa`), and
    - The top semantic suggestions.
  - It fills in **missing categories** in the deterministic tags with the highest-scoring semantic suggestions, up to a small cap (e.g. 1–2 new categories per FOA).
  - The result is a `hybrid_tags` structure that is still ontology-aligned but can cover gaps where the keyword rules are too strict.

This notebook is **exploratory only** and is meant to help iterate on the ontology and understand how deterministic tags compare to embedding-based suggestions for NSF content.

If you run the semantic cells, remember to install `sentence-transformers` and optionally configure `HF_TOKEN` for authenticated model downloads.

---

## Extending / improving the NLP

To tweak or extend the **deterministic tagging** (affecting both the Grants.gov CLI and the NSF notebook’s deterministic path):

- Edit `ontologies.yaml`:
  - Add new categories or tags under `categories:`.
  - For each tag, add more `keywords` entries to cover synonyms and related phrases.
  - Adjust `weight` to emphasize especially diagnostic phrases.
  - Adjust per-category `threshold` if you want more or fewer tags returned.

To experiment with **semantic tagging** for new FOA sources:

- Reuse the pattern from `humanAI_issr4_nsf.ipynb`:
  - Build an ontology label list from `ontologies.yaml`.
  - Encode labels + FOA text with a `SentenceTransformer` model.
  - Compare deterministic vs semantic tags and optionally build hybrid tags for analysis.

