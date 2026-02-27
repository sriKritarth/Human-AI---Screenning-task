import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml


# --------- Utilities ---------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def flatten_tags(tags: Dict[str, Any]) -> Dict[str, str]:
    """
    Flatten nested tag structure into simple CSV-ready columns.
    Input: {category: [{"name": str, "score": float}, ...], ...}
    Output: {"tags_<category>": "name1;name2", ...}
    """
    flat: Dict[str, str] = {}
    if not tags:
        return flat
    for cat, items in tags.items():
        if isinstance(items, list):
            names = [
                x.get("name")
                for x in items
                if isinstance(x, dict) and x.get("name")
            ]
            flat[f"tags_{cat}"] = ";".join(names)
    return flat


# --------- Data model ---------


@dataclass
class FOARecord:
    foa_id: Optional[str] = None
    title: Optional[str] = None
    agency: Optional[str] = None
    posting_date: Optional[str] = None
    close_date: Optional[str] = None
    award_ceiling: Optional[str] = None
    award_floor: Optional[str] = None
    description: Optional[str] = None
    eligibility: Optional[str] = None

    source: str = "grants.gov"
    source_url: str = ""
    retrieved_at: str = ""
    tags: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------- Grants.gov extractor (API-based, deterministic) ---------


class GrantsGovExtractor:
    """
    Minimal wrapper around the public Grants.gov fetchOpportunity endpoint.
    Takes a FOA detail URL and returns a normalized FOARecord.
    """

    FETCH_URL = "https://api.grants.gov/v1/api/fetchOpportunity"

    def __init__(self, url: str, timeout: int = 30) -> None:
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "HumanAI-ISSR4-Screener/1.0",
                "Accept": "application/json,text/html",
            }
        )

    def parse_opportunity_id(self) -> int:
        """
        Example FOA URL:
        https://www.grants.gov/search-results-detail/361087
        """
        m = re.search(r"/search-results-detail/(\d+)", self.url)
        if not m:
            raise ValueError(
                "Unsupported Grants.gov URL. Use a FOA detail URL like: "
                "https://www.grants.gov/search-results-detail/<opportunityId>"
            )
        return int(m.group(1))

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                resp = self.session.post(
                    self.FETCH_URL, json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"fetchOpportunity failed after retries: {last_err}")

    @staticmethod
    def _join_applicant_types(syn: Dict[str, Any]) -> Optional[str]:
        items = syn.get("applicantTypes") or []
        out: List[str] = []
        for it in items:
            if isinstance(it, dict):
                desc = (it.get("description") or "").strip()
                if desc:
                    out.append(desc)
        return "; ".join(out) if out else None

    def extract(self) -> FOARecord:
        opp_id = self.parse_opportunity_id()
        raw = self._post_json({"opportunityId": opp_id})

        data = raw.get("data") or {}
        syn = data.get("synopsis") or {}

        return FOARecord(
            foa_id=str(
                data.get("opportunityNumber") or data.get("id") or opp_id
            ),
            title=data.get("opportunityTitle"),
            agency=syn.get("agencyName") or data.get("owningAgencyCode"),
            posting_date=syn.get("postingDate"),
            close_date=syn.get("closeDate") or syn.get("responseDate"),
            award_ceiling=(
                str(syn.get("awardCeiling"))
                if syn.get("awardCeiling") is not None
                else None
            ),
            award_floor=(
                str(syn.get("awardFloor"))
                if syn.get("awardFloor") is not None
                else None
            ),
            description=syn.get("synopsisDesc"),
            eligibility=self._join_applicant_types(syn),
            source="grants.gov",
            source_url=self.url,
            retrieved_at=utc_now_iso(),
        )


# --------- Lightweight NLP-style, rule-based tagger ---------


@dataclass(frozen=True)
class KeywordRule:
    pattern: str
    weight: float = 1.0


@dataclass(frozen=True)
class OntologyTag:
    name: str
    rules: Tuple[KeywordRule, ...]


class RuleBasedTagger:
    """
    Fully **deterministic**, rule-based tagger implemented as a small ontology.

    - All behavior is encoded in explicit keyword rules with weights.
    - No machine-learning models, randomness, or external services are used.
    - Given the same input text and ontology, it will always return the same tags.
    - Uses light NLP-style preprocessing:
      - lowercasing + whitespace normalization
      - word-boundary-aware matching (avoids matching "health" inside "wealth")
      - multi-keyword rules per tag with weighted scoring
    """

    def __init__(self, ontology_path: str = "ontologies.yaml") -> None:
        """
        Load ontology categories/tags/keyword rules from a YAML file.

        The YAML schema is:

        version: "0.1"
        categories:
          <category_name>:
            threshold: <float>
            tags:
              - name: "<Tag Name>"
                keywords:
                  - { pattern: "<pattern>", weight: <float> }
                  - ...
        """
        self.ontology: Dict[str, Dict[str, Any]] = self._load_ontology(ontology_path)

    @staticmethod
    def _load_ontology(path: str) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Ontology file not found: {path}. "
                "Ensure ontologies.yaml is present in the working directory."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        categories = data.get("categories") or {}
        ontology: Dict[str, Dict[str, Any]] = {}

        for cat_name, cat_cfg in categories.items():
            threshold = float(cat_cfg.get("threshold", 0.35))
            tag_defs = cat_cfg.get("tags") or []
            tags: List[OntologyTag] = []

            for t in tag_defs:
                tag_name = str(t.get("name") or "").strip()
                if not tag_name:
                    continue
                kw_defs = t.get("keywords") or []
                rules: List[KeywordRule] = []
                for kw in kw_defs:
                    pattern = str(kw.get("pattern") or "").strip()
                    if not pattern:
                        continue
                    weight = float(kw.get("weight", 1.0))
                    rules.append(KeywordRule(pattern=pattern, weight=weight))
                if rules:
                    tags.append(OntologyTag(name=tag_name, rules=tuple(rules)))

            ontology[cat_name] = {
                "threshold": threshold,
                "tags": tuple(tags),
            }

        return ontology

    @staticmethod
    def _normalize(text: str) -> str:
        t = (text or "").lower()
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _match_pattern(text: str, pattern: str) -> bool:
        """
        Word-boundary-aware matching to avoid spurious hits, with a small
        plural tolerance for single-word patterns (e.g., "veteran" vs "veterans").
        """
        if not pattern:
            return False

        escaped = re.escape(pattern)

        # Phrase: match whole phrase at word boundaries.
        if " " in pattern:
            regex = rf"\b{escaped}\b"
        else:
            # Single token: allow optional plural "s".
            regex = rf"\b{escaped}s?\b"

        return re.search(regex, text) is not None

    def _score(self, text: str, rules: Tuple[KeywordRule, ...]) -> float:
        total = sum(r.weight for r in rules) or 1.0
        matched = 0.0
        for r in rules:
            if self._match_pattern(text, r.pattern):
                matched += r.weight
        return min(1.0, matched / total)

    def tag(
        self,
        text: str,
        top_k: int = 3,
        keep_all_categories: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Deterministically assign tags by category, then optionally keep only the
        "dominant" semantic category for this FOA.

        Returns (before optional pruning):
            {
                category: [
                    {"name": <tag_name>, "score": 0.0-1.0},
                    ...
                ],
                ...
            }
        """
        txt = self._normalize(text)
        out: Dict[str, List[Dict[str, Any]]] = {}

        for category, cfg in self.ontology.items():
            threshold = float(cfg["threshold"])
            tags: Tuple[OntologyTag, ...] = cfg["tags"]

            scored: List[Dict[str, Any]] = []
            for tag in tags:
                score = self._score(txt, tag.rules)
                if score >= threshold:
                    scored.append(
                        {
                            "name": tag.name,
                            "score": round(score, 3),
                        }
                    )

            scored.sort(key=lambda x: x["score"], reverse=True)
            out[category] = scored[:top_k]

        if keep_all_categories:
            # Keep full multi-category view (may include empty lists),
            # but drop scores from the external representation.
            cleaned: Dict[str, List[Dict[str, Any]]] = {}
            for cat, tags in out.items():
                cleaned[cat] = [{"name": t["name"]} for t in tags]
            return cleaned

        # 1. Drop categories that have no tags above threshold.
        non_empty = {cat: tags for cat, tags in out.items() if tags}
        if not non_empty:
            return out

        # 2. Compute a simple "dominance" signal per category (max tag score).
        cat_max: Dict[str, float] = {
            cat: max(t["score"] for t in tags) for cat, tags in non_empty.items()
        }
        best = max(cat_max.values())

        # 3. Keep only categories whose max score is close to the global best.
        # This approximates "which semantic sector does this FOA mostly live in?"
        dominance_ratio = 0.8
        dominant = {
            cat: non_empty[cat]
            for cat, s in cat_max.items()
            if s >= dominance_ratio * best
        }
        if not dominant:
            return out

        # 4. Within each dominant category, keep only the single highest-score tag.
        # This yields a clear, deterministic "primary" classification per category.
        single_tag_dominant: Dict[str, List[Dict[str, Any]]] = {}
        for cat, tags in dominant.items():
            if not tags:
                continue
            best_tag = max(tags, key=lambda t: t["score"])
            # Expose only the tag name in outputs; scores are used
            # internally for deterministic selection but are not serialized.
            single_tag_dominant[cat] = [{"name": best_tag["name"]}]

        return single_tag_dominant or out


# --------- Export helpers (JSON + CSV) ---------


def write_json(out_dir: str, foa: Dict[str, Any], filename: str = "foa.json") -> str:
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(foa, f, ensure_ascii=False, indent=2)
    return path


def write_csv(out_dir: str, foa: Dict[str, Any], filename: str = "foa.csv") -> str:
    path = os.path.join(out_dir, filename)

    row = dict(foa)
    row.update(flatten_tags(row.get("tags") or {}))

    base_cols = [
        "foa_id",
        "title",
        "agency",
        "posting_date",
        "close_date",
        "award_ceiling",
        "award_floor",
        "description",
        "eligibility",
        "source",
        "source_url",
        "retrieved_at",
    ]
    tag_cols = sorted([k for k in row.keys() if k.startswith("tags_")])
    cols = base_cols + tag_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerow({c: row.get(c) for c in cols})

    return path


# --------- Orchestration / CLI ---------


def detect_source(url: str) -> str:
    """
    Lightweight source detection. Currently we only implement Grants.gov
    but this makes it easy to add NSF later.
    """
    u = url.lower()
    if "grants.gov" in u:
        return "grants.gov"
    if "nsf.gov" in u or "beta.nsf.gov" in u:
        return "nsf"
    raise ValueError("Unsupported URL: expected Grants.gov or NSF FOA detail URL.")


def run_pipeline(url: str, out_dir: str) -> None:
    ensure_out_dir(out_dir)

    source = detect_source(url)

    if source == "grants.gov":
        extractor = GrantsGovExtractor(url)
    else:
        # Skeleton for future NSF support; kept explicit so failures are clear.
        raise ValueError(
            f"NSF URL detected but NSF extractor is not yet implemented: {url}"
        )

    foa = extractor.extract().to_dict()

    # Use multiple text fields for tagging to give the ontology more signal,
    # including the agency so organisation-specific rules can fire.
    text_for_tags_parts: List[str] = []
    for key in ("title", "description", "eligibility", "agency"):
        value = foa.get(key)
        if isinstance(value, str) and value.strip():
            text_for_tags_parts.append(value)
    text_for_tags = " ".join(text_for_tags_parts)

    tagger = RuleBasedTagger()
    foa["tags"] = tagger.tag(text_for_tags)

    write_json(out_dir, foa)
    write_csv(out_dir, foa)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a single FOA URL (Grants.gov), extract core fields, "
            "apply deterministic rule-based NLP tags, and emit foa.json/foa.csv."
        )
    )
    parser.add_argument(
        "--url",
        required=True,
        help="FOA detail URL from Grants.gov (e.g. https://www.grants.gov/search-results-detail/361087)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for foa.json and foa.csv (will be created if missing).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(args.url, args.out_dir)


if __name__ == "__main__":
    main()