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

## NSF notebook: maintainer-facing exploration of a single NSF URL

The notebook `humanAI_issr4_nsf.ipynb` is intentionally structured as an **evaluation notebook**, not just a dump of experiments. It focuses on a **single NSF opportunity URL** and shows maintainers how the problem was approached step by step:

- how the NSF page is ingested,
- how structured FOA fields are extracted,
- how different tagging strategies behave on the same NSF text,
- and why the deterministic approach remains the most task-aligned baseline.

### What the NSF notebook now covers

| Approach | Core idea | Input used from NSF URL | Determinism | Explainability | Handles wording variation | Best use |
|---|---|---|---|---|---|---|
| **1A. Deterministic parser + rule-based ontology tags** | Parse page fields and assign tags from curated ontology rules | Title, description, eligibility, agency | High | High | Low to Medium | Best default baseline for the screening task |
| **1B. Semantic label suggestions** | Compare FOA text against ontology label texts using sentence-transformer embeddings | Same structured FOA text | Medium | Medium | High | Good for recovering tags missed by strict rules |
| **2. TF-IDF baseline** | Match FOA text against ontology labels using classical vector-space similarity | Flattened FOA text and ontology labels | Medium | Medium to High | Medium | Useful lightweight NLP baseline |
| **3. Embedding-based tagging + hybridization** | Use semantic embeddings to rank label candidates, then augment deterministic tags cautiously | Same NSF FOA text plus ontology labels | Medium | Medium | High | Best exploratory approach for broader wording coverage |

### Extraction (NSF)

- Defines an `NSFFOARecord` dataclass similar to `FOARecord`, adapted for NSF opportunity pages.
- Uses `requests` + `BeautifulSoup` to fetch and parse HTML from specific NSF URLs.
- Extracts title, description, and, where available, sections such as eligibility or deadlines directly from the page.

### Approach 1 — deterministic extraction with optional semantic augmentation

This is the **most task-aligned approach** in the notebook.

- `tag_nsf_foa(foa: NSFFOARecord)` wraps the same `RuleBasedTagger` used by the Grants.gov CLI.
- Key fields (`title`, `description`, `eligibility`, `agency`) are concatenated into one text string.
- The ontology in `ontologies.yaml` is applied exactly as in the CLI, so the output `tags` structure remains consistent across Grants.gov and NSF.

An optional semantic extension is then added on top of this baseline:

- ontology labels are embedded with `SentenceTransformer`,
- the FOA text is embedded once,
- cosine similarity is used to rank semantic label suggestions,
- and `augment_tags_with_semantic` fills only selected **missing categories** rather than replacing the deterministic base.

### Approach 2 — TF-IDF baseline

This section adds a classical NLP baseline between pure keyword rules and semantic embeddings.

- The ontology is flattened into label text.
- TF-IDF vectorization is used to compare the NSF FOA text with ontology labels.
- This gives a softer lexical match than exact rules while remaining lighter and more interpretable than full embedding search.

### Approach 3 — embedding-based tagging and hybridization

This section explores whether semantic embeddings improve tag recovery when the NSF page wording differs from ontology keywords.

- Ontology labels are encoded using `sentence-transformers`.
- The FOA text is compared against those label embeddings.
- Top semantic candidates are reviewed as suggestions.
- A cautious hybrid layer is used so the notebook can test better coverage **without discarding explainable deterministic tags**.

### Maintainer note on LLM usage

LLMs were used as a **support tool for understanding the problem, thinking through alternative solution paths, and improving the notebook/readme presentation**. They were **not used as the core tagging engine** for the final deterministic baseline.

To keep the implementation reviewable and aligned with the task:

- the Grants.gov CLI remains fully deterministic,
- the NSF baseline tagging remains ontology-driven,
- the exploratory semantic layer relies on `sentence-transformers` embeddings rather than direct LLM inference.

This notebook is therefore best read as a **maintainer-facing exploration of multiple approaches**, with a clear preference for the deterministic baseline and controlled semantic augmentation where useful.

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



