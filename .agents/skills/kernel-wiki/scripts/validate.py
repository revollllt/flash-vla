#!/usr/bin/env python3
"""Validate YAML frontmatter in all source and wiki pages against schemas,
plus Phase 3 artifact bundles under artifacts/."""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402
from _policy import (  # noqa: E402
    ARCHITECTURE_FAMILY_PREFIXES,
    PRODUCT_ARCHITECTURE_MAPPINGS,
    SUPPORTED_EXACT_ARCHITECTURES,
    body_contract_errors,
)

REPO_ROOT = Path(__file__).parent.parent
SOURCES_DIR = REPO_ROOT / "sources"
WIKI_DIR = REPO_ROOT / "wiki"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CANDIDATES_DIR = REPO_ROOT / "candidates"

REPRO_ORDER = ["concept", "pseudocode", "snippet", "runnable", "benchmarked"]

# This wiki lives inside a repository whose decisions are Agent Notes and whose
# machine constants are the hardware-unit-test skill's table. Both are citable:
# a source id `note-<stem>` resolves to `.agents/notes/**/<stem>.md`, and a
# bracketed `[engine.quantity.scope]` in prose resolves through constants.py.
# A missing sibling degrades to a printed note rather than a failure, so the
# skill stays portable.
import os

NOTES_DIR = Path(os.environ.get("KERNEL_WIKI_NOTES_DIR") or (REPO_ROOT.parents[2] / ".agents" / "notes"))
CONSTANTS_PY = Path(os.environ.get("KERNEL_WIKI_CONSTANTS_PY") or (REPO_ROOT.parent / "hardware-unit-test" / "scripts" / "constants.py"))
KNOWN_PAGE_IDS = set()
MACHINE_TAGS = None
FENCE_RE = re.compile(r"```.*?```", re.S)
MACHINE_TAG_RE = re.compile(r"\[([a-z][a-z0-9]*(?:\.[a-z0-9]+){2,3})\]")
# The sm90 template bundle: pages excerpt its files by name, and its headers
# cite pages and machine constants in brackets. Both directions are checked.
TEMPLATE_BUNDLE = ARTIFACTS_DIR / "kernels" / "sm90-templates" / "variants"
TEMPLATE_FILE_RE = re.compile(r"\b((?:\d\d_[a-z0-9_]+\.cu)|(?:[a-z0-9_]*sm90[a-z0-9_]*\.cuh))\b")
CODE_FENCE_RE = re.compile(r"^```(\S*)\s*\n(.*?)\n```", re.MULTILINE | re.DOTALL)
BRACKET_LIST_RE = re.compile(r"\[([^\]\n]{1,160})\]")
PAGE_ID_RE = re.compile(r"^(?:pattern|technique|hw|kernel|lang|migration)-[a-z0-9]+(?:-[a-z0-9]+)*$")


def _normalize_code(text):
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip("\n")


def excerpt_errors(body, rel):
    """A cpp/cuda fence introduced by a paragraph that names a template file
    must be a contiguous, verbatim excerpt of that file. Fences introduced
    otherwise (an upstream PR excerpt, a sketch) are not held to the bundle."""
    if not TEMPLATE_BUNDLE.is_dir():
        return []
    errors, cursor, cache = [], 0, {}
    for m in CODE_FENCE_RE.finditer(body):
        lang, block = m.group(1).lower(), _normalize_code(m.group(2))
        lead = body[cursor:m.start()].strip()
        cursor = m.end()
        if lang not in ("cpp", "cuda", "c++") or not block:
            continue
        paragraph = lead.split("\n\n")[-1] if lead else ""
        names = sorted(set(TEMPLATE_FILE_RE.findall(paragraph)))
        if not names:
            continue
        for name in names:
            if name not in cache:
                path = TEMPLATE_BUNDLE / name
                cache[name] = _normalize_code(path.read_text(encoding="utf-8")) if path.is_file() else None
        texts = {n: t for n, t in cache.items() if n in names and t is not None}
        if texts and not any(block in text for text in texts.values()):
            first = block.split("\n", 1)[0][:60]
            errors.append(f"{rel}: excerpt starting {first!r} is not found verbatim in {', '.join(texts)}")
    return errors


def template_header_citation_errors():
    """Bracketed citations in template comment lines resolve to pages and tags."""
    errors = []
    if not TEMPLATE_BUNDLE.is_dir():
        return errors
    for path in sorted(TEMPLATE_BUNDLE.glob("*.cu")) + sorted(TEMPLATE_BUNDLE.glob("*.cuh")):
        rel = path.relative_to(REPO_ROOT)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip().startswith("//"):
                continue
            for m in BRACKET_LIST_RE.finditer(line):
                for token in (t.strip() for t in m.group(1).split(",")):
                    if MACHINE_TAG_RE.match(f"[{token}]"):
                        if MACHINE_TAGS is not None and token not in MACHINE_TAGS:
                            errors.append(f"{rel}: [{token}] resolves to no hardware-unit-test constant")
                    elif PAGE_ID_RE.match(token) and token not in KNOWN_PAGE_IDS:
                        errors.append(f"{rel}: [{token}] resolves to no wiki page")
    return errors


def note_source_ids():
    """Agent Notes as citable sources: `note-<stem>` for every active note."""
    ids = set()
    if not NOTES_DIR.is_dir():
        return ids, f"{NOTES_DIR}: no Agent Notes directory; note-* sources unresolvable"
    for lifecycle in ("implemented", "proposed", "rejected"):
        for p in (NOTES_DIR / lifecycle).rglob("*.md"):
            if p.name not in ("README.md", "CLAUDE.md", "AGENTS.md"):
                ids.add(f"note-{p.stem}")
    return ids, None


def load_machine_tags():
    """Ask constants.py, which owns the tag namespace."""
    if not CONSTANTS_PY.is_file():
        return None, f"{CONSTANTS_PY}: not found; machine-constant citations unchecked"
    sys.path.insert(0, str(CONSTANTS_PY.parent))
    try:
        import constants  # noqa: PLC0415
        return {c["tag"] for _, doc in constants.load() for c in doc.get("constants", [])}, None
    except Exception as exc:  # noqa: BLE001
        return None, f"constants.py unusable ({exc}); machine-constant citations unchecked"
    finally:
        sys.path.pop(0)


def machine_tag_citations(body):
    prose = FENCE_RE.sub("", body or "")
    return set(MACHINE_TAG_RE.findall(prose))


def architecture_family_values(family):
    """Return the family token plus every controlled exact target in it."""
    prefixes = ARCHITECTURE_FAMILY_PREFIXES[family]
    return frozenset({
        family,
        *(arch for arch in SUPPORTED_EXACT_ARCHITECTURES if arch.startswith(prefixes)),
    })


HOPPER_ARCHITECTURES = architecture_family_values("hopper")
BLACKWELL_ARCHITECTURES = architecture_family_values("blackwell")

# Offline contract for curated wiki/source-doc links into NVIDIA's rolling PTX
# ISA. Each fragment was resolved against PTX ISA 9.3 on 2026-08-19.
PTX_ISA_CURATED_ANCHORS = frozenset({
    "tensor-memory",
    "tensorcore-5th-generation-instructions",
    "tcgen05-memory-alloc-manage-instructions",
    "parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel",
    "parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel",
})

# Phase 3 per-file 1 MiB cap; bundle 5 MiB cap (see plan AC-10)
FILE_SIZE_CAP_BYTES = 1 * 1024 * 1024
BUNDLE_SIZE_CAP_BYTES = 5 * 1024 * 1024

# Phase 3 source-file extensions that must live in an asset bundle (AC-2).
# `.txt` was added in R23 to cover extract_blog_code.py's unlabeled-fence
# extraction fallback (R20). `.sh`, `.yaml`, `.json` were added in R33 to
# keep this set in sync with extract_blog_code.py's EXT_MAP (the extractor
# emits shell / yaml / json fences into bundles, and orphan + manifest-
# drift detection must cover them too — otherwise a stale `deploy.sh`
# under artifacts/blogs/<slug>/code/ would pass validate.py silently).
# Keep this set identical to the code-ext subset of get_page.py
# --include-code and query.py --has-code; the three together are the
# Phase-3 asset-source contract.
ASSET_SOURCE_EXTS = {
    ".cu", ".cuh", ".ptx",
    ".cpp", ".h", ".hpp",
    ".py", ".pyx",
    ".patch",
    ".inl",
    ".txt",
    ".sh", ".yaml", ".json",
}


def load_yaml_file(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_alias_contract(raw=None):
    """Reject ambiguous aliases and product-to-architecture drift.

    `query.py` resolves the first case-insensitive alias it sees. Without this
    check, a duplicate family/product term can silently canonicalize to the
    wrong exact architecture even when pr_policy's authoritative product map
    is correct.
    """
    errors = []
    if raw is None:
        path = DATA_DIR / "aliases.yaml"
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            return [f"data/aliases.yaml: could not load canonical aliases ({exc})"]
    if not isinstance(raw, dict):
        return ["data/aliases.yaml: top level must be a mapping"]

    resolved = {}
    for canonical, variants in raw.items():
        if not isinstance(canonical, str) or not canonical.strip():
            errors.append("data/aliases.yaml: canonical terms must be non-empty strings")
            continue
        if variants is not None and not isinstance(variants, list):
            errors.append(f"data/aliases.yaml::{canonical}: aliases must be a list")
            continue
        for value in [canonical, *(variants or [])]:
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"data/aliases.yaml::{canonical}: aliases must be non-empty strings"
                )
                continue
            normalized = value.lower()
            previous = resolved.get(normalized)
            if previous is not None and previous != canonical:
                errors.append(
                    f"data/aliases.yaml: {value!r} maps to both {previous!r} "
                    f"and {canonical!r}"
                )
            else:
                resolved[normalized] = canonical

    for product, (expected_architecture, _source) in PRODUCT_ARCHITECTURE_MAPPINGS.items():
        actual = resolved.get(product)
        if actual is not None and actual.lower() != expected_architecture:
            errors.append(
                f"data/aliases.yaml: product {product!r} resolves to {actual!r}; "
                f"pr_policy requires {expected_architecture!r}"
            )
    # Only multi-target generations must remain family-level aliases. A term
    # such as Hopper currently has one canonical base target (sm90), while
    # Blackwell spans sm100/sm103/sm110/sm120/sm121 and cannot collapse.
    for family, exact_prefixes in ARCHITECTURE_FAMILY_PREFIXES.items():
        if len(exact_prefixes) <= 1:
            continue
        actual = resolved.get(family)
        if actual is not None and actual.lower() != family:
            errors.append(
                f"data/aliases.yaml: architecture family {family!r} resolves to "
                f"exact term {actual!r}"
            )
    return errors


def changed_file_inventory_errors(fm, rel="source-pr"):
    """Validate total/listed/displayed changed-file count relationships."""
    errors = []
    total = fm.get("changed_files_count")
    enumerated = fm.get("changed_files_enumerated_count")
    listing_complete = fm.get("changed_files_listing_complete")
    displayed = fm.get("changed_paths") or []
    display_complete = fm.get("changed_paths_complete")
    evidence_count = fm.get("changed_files_evidence_count")
    evidence_complete = fm.get("changed_files_evidence_complete")
    evidence_method = fm.get("changed_files_evidence_method")
    evidence_receipt = fm.get("changed_files_evidence_receipt")

    # Legacy pages may omit the two listing fields. Once either is present,
    # require both so a capped API result cannot masquerade as a total.
    if enumerated is None and listing_complete is None:
        return errors
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        errors.append(f"{rel}: changed_files_count must be a non-negative integer")
        return errors
    if not isinstance(enumerated, int) or isinstance(enumerated, bool) or enumerated < 0:
        errors.append(
            f"{rel}: changed_files_enumerated_count must be a non-negative integer"
        )
        return errors
    if not isinstance(listing_complete, bool):
        errors.append(f"{rel}: changed_files_listing_complete must be boolean")
        return errors
    if enumerated > total:
        errors.append(
            f"{rel}: changed_files_enumerated_count cannot exceed changed_files_count"
        )
    if listing_complete != (enumerated == total):
        errors.append(
            f"{rel}: changed_files_listing_complete must equal "
            "(changed_files_enumerated_count == changed_files_count)"
        )
    evidence_fields = (
        evidence_count, evidence_complete, evidence_method, evidence_receipt
    )
    if any(value is not None for value in evidence_fields):
        if any(value is None for value in evidence_fields):
            errors.append(
                f"{rel}: reconstructed evidence count/completeness/method/receipt "
                "must be supplied together"
            )
        elif (
            not isinstance(evidence_count, int)
            or isinstance(evidence_count, bool)
            or evidence_count < enumerated
            or evidence_count > total
        ):
            errors.append(
                f"{rel}: changed_files_evidence_count must be an integer between "
                "enumerated and total counts"
            )
        else:
            if evidence_complete != (evidence_count == total):
                errors.append(
                    f"{rel}: changed_files_evidence_complete must equal "
                    "(changed_files_evidence_count == changed_files_count)"
                )
            if evidence_method != "github-pull-diff":
                errors.append(
                    f"{rel}: changed_files_evidence_method must be github-pull-diff"
                )
            if evidence_receipt != "audit/pr-file-cap-reconstruction.json":
                errors.append(
                    f"{rel}: changed_files_evidence_receipt must name the committed "
                    "cap reconstruction receipt"
                )
    effective_evidence_count = evidence_count if isinstance(evidence_count, int) else enumerated
    if not isinstance(displayed, list):
        errors.append(f"{rel}: changed_paths must be a list")
    elif len(displayed) > effective_evidence_count:
        errors.append(
            f"{rel}: changed_paths cannot exceed the evaluated changed-file count"
        )
    if display_complete is True and (
        not listing_complete or len(displayed) != total
    ):
        errors.append(
            f"{rel}: changed_paths_complete requires a complete listing and every path"
        )
    return errors


def extract_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    match = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n', content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}


def read_body(filepath):
    """Read the body (post-frontmatter) of a markdown file."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    match = re.match(r'^---\s*\r?\n.*?\r?\n---\s*\r?\n', content, re.DOTALL)
    if match:
        return content[match.end():]
    return content


def detect_page_type(filepath, fm):
    """Detect page type from filepath and frontmatter."""
    rel = filepath.relative_to(REPO_ROOT)
    parts = rel.parts

    if parts[0] == "sources":
        if parts[1] == "prs":
            return "source-pr"
        elif parts[1] == "docs":
            return "source-doc"
        elif parts[1] == "blogs":
            return "source-blog"
        elif parts[1] == "contests":
            return "source-contest"
    elif parts[0] == "wiki":
        t = fm.get("type", "")
        if t:
            return f"wiki-{t}"
        subdir = parts[1] if len(parts) > 1 else ""
        type_map = {
            "hardware": "wiki-hardware",
            "techniques": "wiki-technique",
            "patterns": "wiki-pattern",
            "kernels": "wiki-kernel",
            "languages": "wiki-language",
            "migration": "wiki-migration",
        }
        return type_map.get(subdir, "unknown")
    return "unknown"


def arxiv_source_classification_errors(filepath, fm, page_type):
    """Require arXiv research papers to use the source-doc/paper contract."""
    url = str(fm.get("url") or "")
    if not re.match(r"https?://(?:www\.)?arxiv\.org/", url, re.IGNORECASE):
        return []
    if page_type == "source-doc" and fm.get("source_category") == "paper":
        return []
    try:
        rel = filepath.relative_to(REPO_ROOT)
    except ValueError:
        rel = filepath
    return [
        f"{rel}: arXiv source must use source-doc with source_category: paper "
        f"(got {page_type}, {fm.get('source_category')!r})"
    ]


def merge_sha_contract_errors(fm, rel="source-pr"):
    """Require a full merge SHA exactly for merged PRs."""
    status = fm.get("status")
    has_merge_sha = "merge_sha" in fm
    merge_sha = str(fm.get("merge_sha") or "")
    if status == "merged" and not re.fullmatch(r"[0-9a-fA-F]{40}", merge_sha):
        return [f"{rel}: merged PR requires a full 40-hex merge_sha"]
    if status != "merged" and has_merge_sha:
        return [f"{rel}: merge_sha must be omitted unless status is 'merged'"]
    return []


def ptx_isa_curated_anchor_errors(body, rel="curated page"):
    """Reject unverified PTX ISA fragments in curated source/wiki prose."""
    fragments = set(re.findall(
        r"https://docs\.nvidia\.com/cuda/parallel-thread-execution/"
        r"(?:index\.html)?#([A-Za-z0-9._-]+)",
        body,
    ))
    unknown = sorted(fragments - PTX_ISA_CURATED_ANCHORS)
    return [f"{rel}: unverified PTX ISA fragment '#{fragment}'" for fragment in unknown]


def blackwell_relevance_errors(fm, page_type, rel="wiki page"):
    """Require justification exactly for Hopper-only wiki pages."""
    if not page_type.startswith("wiki-"):
        return []
    archs = set(
        fm.get("architectures", [])
        if isinstance(fm.get("architectures"), list)
        else []
    )
    hopper_archs = archs & HOPPER_ARCHITECTURES
    blackwell_archs = archs & BLACKWELL_ARCHITECTURES
    if hopper_archs and not blackwell_archs and "blackwell_relevance" not in fm:
        return [
            f"{rel}: page targets only Hopper {hopper_archs} without Blackwell arch; "
            "add 'blackwell_relevance' to justify inclusion in Blackwell-first scope"
        ]
    return []


def repro_at_least(level, minimum):
    if level not in REPRO_ORDER or minimum not in REPRO_ORDER:
        return False
    return REPRO_ORDER.index(level) >= REPRO_ORDER.index(minimum)


# Base code languages + all DSLs from data/tags.yaml languages category
_BASE_CODE_LANGS = {
    "cuda", "c", "c++", "cpp", "python", "py", "ptx", "asm",
    "cuda-cpp", "cu", "rust", "shell", "bash", "yaml", "json",
}


def _load_code_langs():
    """Load recognized code fence languages: base set + all from data/tags.yaml."""
    langs = set(_BASE_CODE_LANGS)
    tags_path = DATA_DIR / "tags.yaml"
    if tags_path.exists():
        with open(tags_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        langs.update(data.get("languages", []))
    return langs


# Patterns that indicate real code (not pseudocode or ASCII diagrams)
_CODE_INDICATORS = re.compile(
    # CUDA C++
    r'__global__|__device__|__shared__|__host__|'
    r'asm\s+volatile|#include|#define|#pragma|'
    r'\bvoid\b|\bint\b|\buint32_t\b|\buint64_t\b|\bfloat\b|\bhalf\b|'
    r'\bstruct\b|\btypedef\b|\btemplate\b|\bnamespace\b|\busing\b|\bauto\b|\bconstexpr\b|'
    r'\bfor\s*\(|\bwhile\s*\(|\bif\s*\(|return\s|'
    r'(?:\w+::)+\w+|\w+(?:\.|->)\w+\s*\(|'
    # Python / Triton
    r'\bdef\s+\w+|\bif\s+[^\n:]+:|\bfor\s+\w+\s+in\s+|\breturn\b|'
    r'import\s+\w+|@triton\.jit|tl\.\w+|\w+\s*=\s*\w+\s*\(|'
    # Build / shell snippets
    r'\bnvcc\s|\bcmake\s|\bpython3?\s|'
    # PTX
    r'tcgen05|mbarrier|cp\.async|ld\.global|st\.global|'
    r'\.reg\s|\.pred\s|cvt\.\w+|mov\.b32|'
    # TileLang (TVM-based DSL)
    r'@T\.prim_func|T\.alloc_buffer|T\.grid|T\.block_attr|T\.reads|T\.writes|'
    # cuTile (NVIDIA Python DSL)
    r'cutile\.\w+|@cutile\.kernel|tile_load|tile_store|tile_mma|'
    # JAX Pallas
    r'pl\.\w+|@pl\.kernel|pallas\.|jax\.\w+|jnp\.\w+'
)


def has_compilable_code(body, code_langs):
    """Check if body contains a fenced code block with a known language, real code,
    and at least 2 non-blank non-comment lines (rejects one-line stubs)."""
    for m in re.finditer(r'^```(\S*)\s*\n(.*?)\n```', body, re.MULTILINE | re.DOTALL):
        info = m.group(1).lower()
        block = m.group(2)
        if info not in code_langs:
            continue
        if not _CODE_INDICATORS.search(block):
            continue
        # Count substantive code lines (not blank, not comment-only)
        code_lines = 0
        for line in block.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                code_lines += 1
        if code_lines >= 2:
            return True
    return False


def validate_file(filepath, schemas, valid_tags, all_source_ids, code_langs):
    """Validate a single file. Returns list of error strings."""
    errors = []
    rel = filepath.relative_to(REPO_ROOT)

    fm = extract_frontmatter(filepath)
    if fm is None:
        errors.append(f"{rel}: missing YAML frontmatter")
        return errors
    if not isinstance(fm, dict):
        errors.append(f"{rel}: frontmatter must be a YAML mapping, got {type(fm).__name__}")
        return errors
    if "_parse_error" in fm:
        errors.append(f"{rel}: YAML parse error: {fm['_parse_error']}")
        return errors

    page_type = detect_page_type(filepath, fm)
    if page_type == "unknown":
        errors.append(f"{rel}: unknown page type")
        return errors

    errors.extend(arxiv_source_classification_errors(filepath, fm, page_type))
    if page_type == "source-pr":
        errors.extend(merge_sha_contract_errors(fm, rel))
    if page_type.startswith("wiki-") or page_type == "source-doc":
        errors.extend(ptx_isa_curated_anchor_errors(read_body(filepath), rel))

    schema = schemas.get(page_type)
    if not schema:
        errors.append(f"{rel}: no schema defined for type '{page_type}'")
        return errors

    constraints = schema.get("constraints", {})

    # Check required fields
    for field in schema.get("required", []):
        if field not in fm or fm[field] is None:
            errors.append(f"{rel}: missing required field '{field}'")

    # Validate id_prefix
    id_prefix = constraints.get("id_prefix")
    if id_prefix and "id" in fm:
        if not str(fm["id"]).startswith(id_prefix):
            errors.append(f"{rel}: id '{fm['id']}' must start with '{id_prefix}'")

    # Build per-field vocabulary sets
    # "tags" accepts only topical categories (not architectures/confidence/etc.)
    topical_categories = ["hardware_features", "techniques", "kernel_types", "languages"]
    tags_valid = set()
    for cat in topical_categories:
        tags_valid.update(valid_tags.get(cat, []))

    field_vocab = {
        "tags": tags_valid,
        "techniques": set(valid_tags.get("techniques", [])),
        "hardware_features": set(valid_tags.get("hardware_features", [])),
        "kernel_types": set(valid_tags.get("kernel_types", [])),
        "languages": set(valid_tags.get("languages", [])),
    }

    # Check list type and uniqueness for all list-valued fields
    list_fields = ["tags", "techniques", "hardware_features", "kernel_types", "languages",
                    "architectures", "related", "sources", "symptoms", "candidate_techniques",
                    "prerequisites", "aliases"]
    for tag_field in list_fields:
        if tag_field in fm:
            if not isinstance(fm[tag_field], list):
                errors.append(f"{rel}: field '{tag_field}' must be a YAML list, got {type(fm[tag_field]).__name__}")
                continue
            # Reject duplicates
            seen = set()
            for val in fm[tag_field]:
                if val in seen:
                    errors.append(f"{rel}: duplicate value '{val}' in field '{tag_field}'")
                seen.add(val)

    # Check hardware tags are reflected in hardware_features
    if "hardware_features" in fm and isinstance(fm["hardware_features"], list):
        hw_in_tags = set(fm.get("tags", [])) & set(field_vocab.get("hardware_features", set()))
        hw_explicit = set(fm["hardware_features"])
        missing_hw = hw_in_tags - hw_explicit
        if missing_hw:
            errors.append(
                f"{rel}: tags contain hardware features {sorted(missing_hw)} "
                f"not in hardware_features field"
            )

    # Validate each structured field against its own vocabulary
    for tag_field, vocab in field_vocab.items():
        if tag_field in fm and isinstance(fm[tag_field], list):
            for tag in fm[tag_field]:
                if tag not in vocab:
                    errors.append(f"{rel}: '{tag}' is not a valid {tag_field} value")

    # Validate candidate_techniques entries are known page ID prefixes
    valid_remedy_prefixes = ("technique-", "hw-", "migration-")
    if "candidate_techniques" in fm and isinstance(fm["candidate_techniques"], list):
        for ct in fm["candidate_techniques"]:
            if not str(ct).startswith(valid_remedy_prefixes):
                errors.append(
                    f"{rel}: candidate_techniques entry '{ct}' must use one of "
                    f"{valid_remedy_prefixes} prefixes"
                )

    # Validate architectures
    valid_archs = set(valid_tags.get("architectures", []))
    if "architectures" in fm and isinstance(fm["architectures"], list):
        for arch in fm["architectures"]:
            if arch not in valid_archs:
                errors.append(f"{rel}: unknown architecture '{arch}'")

    # Factual-audit architecture contract for generated PR sources.  Empty is
    # permitted only as an explicit, visible, evidence-noted unknown.
    if page_type == "source-pr":
        errors.extend(changed_file_inventory_errors(fm, str(rel)))
        archs = fm.get("architectures")
        disposition = fm.get("architecture_disposition")
        evidence = fm.get("architecture_evidence")
        if disposition not in {"exact", "family", "mixed", "unknown"}:
            errors.append(
                f"{rel}: architecture_disposition must be exact, family, mixed, or unknown"
            )
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{rel}: architecture_evidence must be a non-empty list")
            evidence = []
        evidence_values = set()
        for index, row in enumerate(evidence):
            if not isinstance(row, dict):
                errors.append(f"{rel}: architecture_evidence[{index}] must be a mapping")
                continue
            for field in ("architecture", "basis", "locator", "evidence"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    errors.append(
                        f"{rel}: architecture_evidence[{index}].{field} must be a non-empty string"
                    )
            if row.get("architecture"):
                evidence_values.add(row["architecture"])
            if row.get("basis") == "documented-product-mapping":
                mapping_source = row.get("mapping_source")
                if not isinstance(mapping_source, str) or not mapping_source.startswith("https://"):
                    errors.append(
                        f"{rel}: documented product mapping requires an HTTPS mapping_source"
                    )
        if isinstance(archs, list):
            if disposition == "unknown":
                if archs:
                    errors.append(f"{rel}: unknown disposition requires architectures: []")
                if "unknown" not in evidence_values:
                    errors.append(f"{rel}: unknown disposition requires an unknown evidence note")
            elif disposition == "family":
                family_values = set(ARCHITECTURE_FAMILY_PREFIXES)
                if not archs or not set(archs).issubset(family_values):
                    errors.append(
                        f"{rel}: family disposition requires only recognized family values"
                    )
                unsupported = set(archs or []) - evidence_values
                if unsupported:
                    errors.append(f"{rel}: family assignments lack matching evidence: {sorted(unsupported)}")
            elif disposition == "exact":
                family_values = set(ARCHITECTURE_FAMILY_PREFIXES)
                if not archs or set(archs) & family_values:
                    errors.append(
                        f"{rel}: exact disposition requires one or more exact values and no family placeholder"
                    )
                unsupported = set(archs or []) - evidence_values
                if unsupported:
                    errors.append(
                        f"{rel}: architecture assignments lack matching evidence: {sorted(unsupported)}"
                    )
            elif disposition == "mixed":
                family_values = set(ARCHITECTURE_FAMILY_PREFIXES)
                values = set(archs or [])
                if not (values & family_values) or not (values - family_values):
                    errors.append(f"{rel}: mixed disposition requires family and exact values")
                unsupported = values - evidence_values
                if unsupported:
                    errors.append(
                        f"{rel}: mixed architecture assignments lack matching evidence: {sorted(unsupported)}"
                    )

        scope_disposition = fm.get("scope_disposition")
        scope_evidence = fm.get("scope_evidence")
        if scope_disposition != "retained":
            errors.append(f"{rel}: source PR pages must have scope_disposition: retained")
        allowed_scope_rules = {
            "cuda-cute-ptx-device-source",
            "python-dsl-device-kernel",
            "device-code-signal",
        }
        if not isinstance(scope_evidence, dict):
            errors.append(f"{rel}: scope_evidence must be a mapping")
        else:
            rule = scope_evidence.get("rule")
            paths = scope_evidence.get("paths")
            if rule not in allowed_scope_rules:
                errors.append(f"{rel}: unsupported retained scope rule {rule!r}")
            if not isinstance(paths, list) or not paths:
                errors.append(f"{rel}: retained scope_evidence.paths must be a non-empty list")
            else:
                displayed_paths = set(fm.get("changed_paths") or [])
                absent = set(paths) - displayed_paths
                if absent:
                    errors.append(
                        f"{rel}: scope evidence paths absent from changed_paths: {sorted(absent)}"
                    )
                if rule == "cuda-cute-ptx-device-source" and any(
                    Path(path).suffix.lower() not in {".cu", ".cuh", ".ptx"} for path in paths
                ):
                    errors.append(f"{rel}: CUDA/CuTe/PTX scope rule cites a non-device-source path")
                if rule == "python-dsl-device-kernel" and any(
                    Path(path).suffix.lower() != ".py" for path in paths
                ):
                    errors.append(f"{rel}: Python DSL scope rule cites a non-Python path")
        inclusion_reason = fm.get("inclusion_reason")
        if not isinstance(inclusion_reason, str) or not inclusion_reason.startswith("retain:"):
            errors.append(f"{rel}: inclusion_reason must name the concrete retained rule/evidence")
        # Named distributed-system exclusions are hard when they are the PR's
        # subject. A qualifying single-device kernel PR may still touch a
        # DeepEP path or mention a backend in a benchmark/test; the shared
        # intake policy excludes those paths from its positive evidence.
        hard_scope_title = str(fm.get("title", ""))
        if re.search(r"(?i)(?<![a-z0-9])(eplb|deep[ _-]?ep|dual[ _-]?pipe)(?![a-z0-9])", hard_scope_title):
            errors.append(f"{rel}: hard-excluded EPLB/DeepEP/DualPipe source page is present")

        locator = fm.get("upstream_body_locator")
        if not isinstance(locator, str) or not locator.startswith("https://"):
            errors.append(f"{rel}: upstream_body_locator must be a non-empty HTTPS locator")
        for problem in body_contract_errors(fm, read_body(filepath)):
            errors.append(f"{rel}: {problem}")

    # Validate from_arch / to_arch on migration pages
    for arch_field in ["from_arch", "to_arch"]:
        if arch_field in fm:
            if fm[arch_field] not in valid_archs:
                errors.append(f"{rel}: {arch_field} '{fm[arch_field]}' is not a known architecture")

    # Validate confidence
    valid_conf = set(valid_tags.get("confidence", []))
    if "confidence" in fm and fm["confidence"] not in valid_conf:
        errors.append(f"{rel}: invalid confidence '{fm['confidence']}'")

    # Validate reproducibility
    valid_repro = set(valid_tags.get("reproducibility", []))
    if "reproducibility" in fm:
        if fm["reproducibility"] not in valid_repro:
            errors.append(f"{rel}: invalid reproducibility '{fm['reproducibility']}'")

    # Check reproducibility minimum
    repro_min = constraints.get("reproducibility_minimum")
    if repro_min and "reproducibility" in fm:
        if not repro_at_least(fm["reproducibility"], repro_min):
            errors.append(
                f"{rel}: reproducibility '{fm['reproducibility']}' below "
                f"minimum '{repro_min}' for {page_type}"
            )

    # Validate source_category against schema constraints
    valid_cats = set(valid_tags.get("source_categories", []))
    if "source_category" in fm:
        cat = fm["source_category"]
        if cat not in valid_cats:
            errors.append(f"{rel}: invalid source_category '{cat}'")
        # Check schema-specific category constraints
        cat_constraint = constraints.get("source_category")
        if cat_constraint:
            allowed = cat_constraint if isinstance(cat_constraint, list) else [cat_constraint]
            if cat not in allowed:
                errors.append(f"{rel}: source_category '{cat}' not in allowed {allowed}")

    # Validate status enum
    status_constraint = constraints.get("status")
    if status_constraint and "status" in fm:
        allowed = status_constraint if isinstance(status_constraint, list) else [status_constraint]
        if fm["status"] not in allowed:
            errors.append(f"{rel}: status '{fm['status']}' not in {allowed}")

    # Check type field matches constraint
    if "type" in constraints and "type" in fm:
        if fm["type"] != constraints["type"]:
            errors.append(
                f"{rel}: type '{fm['type']}' does not match "
                f"expected '{constraints['type']}' for {page_type}"
            )

    # Check performance_claims structure (including shape and numeric value)
    if "performance_claims" in fm:
        pc = fm["performance_claims"]
        if not isinstance(pc, list):
            errors.append(f"{rel}: performance_claims must be a list, got {type(pc).__name__}")
        else:
            for i, claim in enumerate(pc):
                if not isinstance(claim, dict):
                    errors.append(f"{rel}: performance_claims[{i}] must be a mapping, got {type(claim).__name__}")
                    continue
                for req in ["gpu", "dtype", "shape", "metric", "value", "source_id", "source_locator"]:
                    if req not in claim:
                        errors.append(f"{rel}: performance_claims[{i}] missing '{req}'")
                if "value" in claim and not isinstance(claim["value"], (int, float)):
                    errors.append(
                        f"{rel}: performance_claims[{i}].value must be numeric, "
                        f"got {type(claim['value']).__name__}: {claim['value']}"
                    )
                # Cross-check source_id against known source IDs
                sid = claim.get("source_id", "")
                if sid and all_source_ids and sid not in all_source_ids:
                    errors.append(
                        f"{rel}: performance_claims[{i}].source_id '{sid}' "
                        f"not found in source corpus"
                    )
                locator = claim.get("source_locator")
                if "source_locator" in claim and (
                    not isinstance(locator, str) or not locator.strip()
                ):
                    errors.append(
                        f"{rel}: performance_claims[{i}].source_locator must be a non-empty exact locator"
                    )

    # Check wiki sources reference existing source ids
    if page_type.startswith("wiki-") and "sources" in fm and isinstance(fm["sources"], list):
        for src_id in fm["sources"]:
            if all_source_ids and src_id not in all_source_ids:
                errors.append(f"{rel}: references unknown source id '{src_id}'")

    # AC-9: Enforce evidence_basis for verified wiki pages
    if page_type.startswith("wiki-") and fm.get("confidence") == "verified":
        eb = fm.get("evidence_basis")
        if not eb or not isinstance(eb, list) or len(eb) == 0:
            errors.append(
                f"{rel}: confidence 'verified' requires non-empty 'evidence_basis' field"
            )
        else:
            eb_types = {entry.get("evidence_type") for entry in eb if isinstance(entry, dict)}
            if "official-doc" not in eb_types:
                errors.append(
                    f"{rel}: evidence_basis for 'verified' must include at least one "
                    f"'official-doc' entry (found: {eb_types})"
                )
            if "upstream-code" not in eb_types:
                errors.append(
                    f"{rel}: evidence_basis for 'verified' must include at least one "
                    f"'upstream-code' entry (found: {eb_types})"
                )
            # Cross-check evidence_basis source_ids against page sources
            page_sources = set(fm.get("sources", []))
            for entry in eb:
                if isinstance(entry, dict):
                    sid = entry.get("source_id", "")
                    if sid and sid not in page_sources:
                        errors.append(
                            f"{rel}: evidence_basis references '{sid}' "
                            f"not listed in page sources"
                        )

    # `measured` is this wiki's first-hand level: the page cites the benchmark or
    # reproduction that established it, the same way `verified` cites a doc and
    # upstream code. Evidence lives in a hardware-unit-test constant, a
    # reference-grade template's STATUS block, or an Agent Note.
    if page_type.startswith("wiki-") and fm.get("confidence") == "measured":
        eb = fm.get("evidence_basis")
        if not eb or not isinstance(eb, list):
            errors.append(f"{rel}: confidence 'measured' requires non-empty 'evidence_basis' field")
        else:
            eb_types = {entry.get("evidence_type") for entry in eb if isinstance(entry, dict)}
            if not eb_types & {"benchmark", "reproduction"}:
                errors.append(
                    f"{rel}: evidence_basis for 'measured' must include a 'benchmark' or "
                    f"'reproduction' entry (found: {eb_types})"
                )
            page_sources = set(fm.get("sources", []))
            for entry in eb:
                if isinstance(entry, dict):
                    sid = entry.get("source_id", "")
                    if sid and sid not in page_sources:
                        errors.append(f"{rel}: evidence_basis references '{sid}' not listed in page sources")

    # Symptoms come from the controlled list so the by-problem index and the
    # profiler's vocabulary stay one language.
    valid_symptoms = set(valid_tags.get("symptoms", []))
    if "symptoms" in fm and isinstance(fm["symptoms"], list) and valid_symptoms:
        for s in fm["symptoms"]:
            if s not in valid_symptoms:
                errors.append(f"{rel}: '{s}' is not a valid symptoms value")

    # Cross-references to other pages must resolve; a dangling id is a dead link
    # the reader cannot follow.
    if page_type.startswith("wiki-") and KNOWN_PAGE_IDS:
        for field in ("related", "prerequisites", "candidate_techniques"):
            for pid in fm.get(field) or []:
                if pid not in KNOWN_PAGE_IDS:
                    errors.append(f"{rel}: {field} entry '{pid}' does not resolve to a page id")

    # Relative markdown links must resolve on disk.
    body_for_links = read_body(filepath)
    for m in re.finditer(r"\]\(([^)\s#]+)(?:#[^)]*)?\)", body_for_links):
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:")) or not ("/" in target or target.endswith(".md")):
            continue
        if not (filepath.parent / target).resolve().exists():
            errors.append(f"{rel}: relative link '{target}' does not resolve")

    # Machine constants are cited in brackets by hardware-unit-test tag and
    # never restated; every tag must resolve to a constant in that table.
    if page_type.startswith("wiki-") and MACHINE_TAGS is not None:
        for tag in machine_tag_citations(body_for_links):
            if tag not in MACHINE_TAGS:
                errors.append(f"{rel}: machine-constant citation [{tag}] resolves to no hardware-unit-test constant")

    if page_type.startswith("wiki-"):
        errors.extend(excerpt_errors(body_for_links, rel))

    # Pages that claim snippet-or-better reproducibility must actually carry a
    # compilable fence. Concept and pseudocode pages are allowed to remain
    # honest prose instead of being forced to invent an implementation.
    if (
        page_type in ("wiki-technique", "wiki-kernel", "wiki-language")
        and repro_at_least(fm.get("reproducibility", "concept"), "snippet")
    ):
        body = read_body(filepath)
        if not has_compilable_code(body, code_langs):
            errors.append(f"{rel}: {page_type} page must contain fenced code block (reproducibility >= snippet)")

    # Phase 3 AC-11: enforce disallow_peer_of_artifact_dir at page top level
    disallow = schemas.get(page_type, {}).get("disallow_peer_of_artifact_dir") or []
    for banned in disallow:
        if banned in fm:
            errors.append(
                f"{rel}: page frontmatter must not carry '{banned}:' at top level "
                f"(single-field contract: use 'artifact_dir:' only; per-file pointers "
                f"live inside PROVENANCE.yaml)"
            )

    # Phase 3 AC-5/AC-11: if artifact_dir is set, it must resolve to a real directory
    if "artifact_dir" in fm:
        ad = fm["artifact_dir"]
        if not isinstance(ad, str):
            errors.append(f"{rel}: artifact_dir must be a string path")
        else:
            target = REPO_ROOT / ad
            if not target.is_dir():
                errors.append(f"{rel}: artifact_dir '{ad}' does not resolve to an existing directory")
            else:
                # R34: resolve symlinks / `..` traversal before checking
                # containment under REPO_ROOT/artifacts. A raw
                # startswith('artifacts/') check accepts strings like
                # 'artifacts/../sources/prs' that escape the quarantine.
                # get_page.py --include-code and query.py --has-code
                # both follow the resolved path, so the validator must
                # compare resolved-vs-resolved too.
                try:
                    resolved = target.resolve()
                    artifacts_root = (REPO_ROOT / "artifacts").resolve()
                    if resolved != artifacts_root and artifacts_root not in resolved.parents:
                        errors.append(
                            f"{rel}: artifact_dir '{ad}' must live under 'artifacts/' "
                            f"(code assets are quarantined from wiki/ per AC-9); "
                            f"resolves to '{resolved}' which is outside "
                            f"'{artifacts_root}'"
                        )
                except (OSError, RuntimeError) as e:
                    errors.append(f"{rel}: artifact_dir '{ad}' could not be resolved: {e}")

    # Phase 3 AC-3: nested submissions[*] validation on source-contest pages
    if page_type == "source-contest" and "submissions" in fm:
        errors.extend(validate_contest_submissions(fm, filepath, schemas))

    return errors


def validate_contest_submissions(fm, filepath, schemas):
    """AC-3: enforce truth-model enum, conditional code_path / reason, and
    contest-bundle containment (code_path must resolve inside the page's own
    implicit submission bundle, not just anywhere under artifacts/contests/)."""
    rel = filepath.relative_to(REPO_ROOT)
    errors = []
    subs = fm.get("submissions") or []
    sub_schema = schemas.get("source-contest", {}).get("submissions_schema", {})
    required = sub_schema.get("required", [])
    optional = sub_schema.get("optional", [])
    allowed_truths = (sub_schema.get("constraints") or {}).get("submission_truth") or []

    # Implicit submission bundle root: artifacts/contests/<contest>/<problem>/submissions/
    # where <contest> is the page's parent directory name and <problem> is the
    # page's filename stem.
    contest_slug = filepath.parent.name
    problem_slug = filepath.stem
    expected_prefix = f"artifacts/contests/{contest_slug}/{problem_slug}/submissions/"
    # R34: resolve-vs-resolve containment check. A raw startswith
    # would accept a `code_path` like `artifacts/contests/<c>/<p>/
    # submissions/../rank-2-other/file.cpp` which escapes the
    # submission bundle for this row.
    try:
        expected_root = (REPO_ROOT / expected_prefix).resolve()
    except (OSError, RuntimeError):
        expected_root = REPO_ROOT / expected_prefix

    for i, entry in enumerate(subs):
        if not isinstance(entry, dict):
            errors.append(f"{rel}: submissions[{i}] must be a mapping, got {type(entry).__name__}")
            continue
        for req in required:
            if req not in entry:
                errors.append(f"{rel}: submissions[{i}] missing required '{req}'")
        # Enum
        truth = entry.get("submission_truth")
        if truth is not None and allowed_truths and truth not in allowed_truths:
            errors.append(f"{rel}: submissions[{i}].submission_truth '{truth}' not in {allowed_truths}")
        # Conditional requirements
        if truth == "unavailable":
            if not entry.get("code_unavailable_reason"):
                errors.append(
                    f"{rel}: submissions[{i}] has submission_truth='unavailable' "
                    f"but no 'code_unavailable_reason'"
                )
        elif truth is not None:
            cp = entry.get("code_path")
            if not cp:
                errors.append(
                    f"{rel}: submissions[{i}] has submission_truth='{truth}' but no 'code_path'"
                )
            else:
                target = REPO_ROOT / cp
                if not target.exists():
                    errors.append(
                        f"{rel}: submissions[{i}].code_path '{cp}' does not exist"
                    )
                else:
                    # R34: compare RESOLVED paths so `..` traversal is
                    # caught. A raw startswith string check accepts
                    # `.../submissions/../rank-2-other/...` which
                    # points outside this row's submission bundle.
                    try:
                        resolved_cp = target.resolve()
                        if expected_root not in resolved_cp.parents and resolved_cp != expected_root:
                            errors.append(
                                f"{rel}: submissions[{i}].code_path '{cp}' must live under "
                                f"'{expected_prefix}' (the page's own implicit submission bundle), "
                                f"not an arbitrary location inside artifacts/contests/; "
                                f"resolves to '{resolved_cp}' which is outside "
                                f"'{expected_root}'"
                            )
                    except (OSError, RuntimeError) as e:
                        errors.append(
                            f"{rel}: submissions[{i}].code_path '{cp}' could not be resolved: {e}"
                        )
        # Reject unknown fields strictly
        allowed_fields = set(required) | set(optional)
        for k in entry.keys():
            if k not in allowed_fields:
                errors.append(f"{rel}: submissions[{i}] has unknown field '{k}'")
    return errors


# ---------------------------------------------------------------------------
# Phase 3: artifact bundle validation (AC-2, AC-5, AC-9, AC-10, AC-11)
# ---------------------------------------------------------------------------

## Ledger-shape check (AC-3 from plan-phase4.md). Every candidates/*.yaml
## must carry the canonical top-level fields and its summary counts must
## match its row decisions.
LEDGER_REQUIRED_TOP_FIELDS = [
    "repo",
    "searched_at",
    "keywords_used",
    "total_candidates",
    "included",
    "excluded",
    "deferred",
    "prs",
]
LEDGER_REQUIRED_PR_FIELDS = ["number", "title", "date", "decision", "reason"]


def validate_ledger(ledger_path):
    """Validate a candidate ledger file's top-level shape and summary
    consistency. Returns a list of error strings (empty if valid)."""
    errors = []
    rel = ledger_path.relative_to(REPO_ROOT)
    try:
        data = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return [f"{rel}: invalid YAML ({e})"]
    if not isinstance(data, dict):
        return [f"{rel}: top-level value must be a mapping"]
    for field in LEDGER_REQUIRED_TOP_FIELDS:
        if field not in data:
            errors.append(f"{rel}: missing required top-level field '{field}'")
    if errors:
        # Don't continue with summary-count check if shape is broken.
        return errors
    prs = data["prs"]
    if not isinstance(prs, list):
        return [f"{rel}: 'prs' must be a list, got {type(prs).__name__}"]
    inc = exc = dfr = 0
    for i, row in enumerate(prs):
        if not isinstance(row, dict):
            errors.append(f"{rel}: prs[{i}] must be a mapping")
            continue
        for f in LEDGER_REQUIRED_PR_FIELDS:
            if f not in row:
                errors.append(f"{rel}: prs[{i}] missing required field '{f}'")
        d = str(row.get("decision", "")).lower()
        if d == "include":
            inc += 1
        elif d == "exclude":
            exc += 1
        elif d == "defer":
            dfr += 1
        else:
            errors.append(f"{rel}: prs[{i}] has unknown decision '{row.get('decision')}'")
    if data["total_candidates"] != len(prs):
        errors.append(
            f"{rel}: total_candidates={data['total_candidates']} disagrees with len(prs)={len(prs)}"
        )
    if data["included"] != inc:
        errors.append(f"{rel}: included={data['included']} disagrees with row count {inc}")
    if data["excluded"] != exc:
        errors.append(f"{rel}: excluded={data['excluded']} disagrees with row count {exc}")
    if data["deferred"] != dfr:
        errors.append(f"{rel}: deferred={data['deferred']} disagrees with row count {dfr}")
    if (
        data["total_candidates"] != data["included"] + data["excluded"] + data["deferred"]
    ):
        errors.append(
            f"{rel}: total_candidates ({data['total_candidates']}) != "
            f"included + excluded + deferred ("
            f"{data['included']} + {data['excluded']} + {data['deferred']})"
        )
    return errors


## AC-2 hybrid-registry presence check (DEC-1). Pages in scope that carry a
## per-page `version_sensitive: <id>` pointer must resolve to a claim in
## `data/version-claims.yaml`; reverse direction is also enforced —
## every registry entry's `applies_to` paths must exist.
##
## Scope is exactly: wiki/**/*.md, references/primer.md, references/examples.md,
## and parsed YAML scalars data/inclusion-policy.yaml::{cute-dsl,triton}.description.
def validate_version_claims_registry(all_source_ids):
    """Return list of error strings for AC-2 hybrid-registry consistency."""
    errors = []
    claims_path = DATA_DIR / "version-claims.yaml"
    if not claims_path.is_file():
        return ["data/version-claims.yaml: missing (DEC-1 hybrid registry stub required)"]
    try:
        data = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return [f"data/version-claims.yaml: invalid YAML ({e})"]
    claims = (data or {}).get("claims") or []

    # Build registry-id -> applies_to mapping for reverse-direction check.
    claim_by_id = {}
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"data/version-claims.yaml::claims[{i}]: must be a mapping")
            continue
        cid = claim.get("id")
        if not cid:
            errors.append(f"data/version-claims.yaml::claims[{i}]: missing id")
            continue
        claim_by_id[cid] = claim
        # source_ids resolution
        for sid in claim.get("source_ids", []) or []:
            if sid not in all_source_ids:
                errors.append(f"data/version-claims.yaml::{cid}: source_id '{sid}' does not resolve")
        # applies_to: every target must EXIST and CARRY the matching pointer.
        # Reverse direction enforcement (the AC-2 strict check Codex Round 3
        # required): existence alone is insufficient.
        for applies in claim.get("applies_to", []) or []:
            file_part, _, scalar_pointer = applies.partition("::")
            target_path = REPO_ROOT / file_part
            if not target_path.exists():
                errors.append(f"data/version-claims.yaml::{cid}: applies_to '{applies}' (file '{file_part}') does not exist")
                continue
            if scalar_pointer:
                # YAML JSON-pointer form (e.g. data/inclusion-policy.yaml::triton.description).
                # The reverse-direction proof for these is the existence of
                # an authoring rule recorded elsewhere; we don't try to rewrite
                # YAML scalars to embed a pointer. The `applies_to` path itself
                # is the authoring-time anchor.
                continue
            # Markdown file: must carry version_sensitive: <id> in frontmatter.
            try:
                fm = extract_frontmatter(target_path)
            except Exception as e:
                errors.append(f"data/version-claims.yaml::{cid}: applies_to '{applies}' frontmatter parse failed: {e}")
                continue
            if not fm or not isinstance(fm, dict):
                errors.append(f"data/version-claims.yaml::{cid}: applies_to '{applies}' has no frontmatter (AC-2 reverse direction: target must carry version_sensitive: {cid})")
                continue
            vs = fm.get("version_sensitive")
            ptr_id = vs.get("id") if isinstance(vs, dict) else vs
            if ptr_id != cid:
                errors.append(f"data/version-claims.yaml::{cid}: applies_to '{applies}' carries version_sensitive id={ptr_id!r}, expected {cid!r} (AC-2 reverse direction)")

    # Forward direction: every page in scope with a per-page pointer must
    # resolve to a registry entry. Pages without a pointer are not flagged
    # here — flag-on-missing-pointer is the job of the AC-2 surface check
    # below (parsed YAML scalar detection).
    in_scope = []
    if WIKI_DIR.exists():
        in_scope.extend(sorted(WIKI_DIR.rglob("*.md")))
    for ref in (REPO_ROOT / "references" / "primer.md", REPO_ROOT / "references" / "examples.md"):
        if ref.is_file():
            in_scope.append(ref)
    for md_file in in_scope:
        fm = extract_frontmatter(md_file)
        if not fm or not isinstance(fm, dict):
            continue
        vs = fm.get("version_sensitive")
        if vs is None:
            continue
        # vs may be a dict {id: ...} or a string id
        ptr = vs.get("id") if isinstance(vs, dict) else vs
        if not ptr:
            errors.append(f"{md_file.relative_to(REPO_ROOT)}: version_sensitive block has no id")
            continue
        if ptr not in claim_by_id:
            errors.append(f"{md_file.relative_to(REPO_ROOT)}: version_sensitive id '{ptr}' does not resolve to data/version-claims.yaml")

    return errors


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_bundle_roots():
    """Yield bundle-root Paths — directories under artifacts/ that match the
    standard layout (plan AC-2).

    Standard layout:
      artifacts/prs/<repo>/PR-<N>/
      artifacts/contests/<contest>/<problem>/submissions/<rank-N-author>/
      artifacts/blogs/<slug>/code/
      artifacts/kernels/<slug>/full/
      artifacts/kernels/<slug>/variants/
    """
    if not ARTIFACTS_DIR.is_dir():
        return
    # PR bundles
    prs = ARTIFACTS_DIR / "prs"
    if prs.is_dir():
        for repo in sorted(prs.iterdir()):
            if repo.is_dir():
                for pr_dir in sorted(repo.iterdir()):
                    if pr_dir.is_dir() and pr_dir.name.startswith("PR-"):
                        yield pr_dir
    # Contest submissions
    contests = ARTIFACTS_DIR / "contests"
    if contests.is_dir():
        for contest in sorted(contests.iterdir()):
            if contest.is_dir():
                for problem in sorted(contest.iterdir()):
                    if problem.is_dir():
                        subs = problem / "submissions"
                        if subs.is_dir():
                            for sub in sorted(subs.iterdir()):
                                if sub.is_dir():
                                    yield sub
    # Blog code
    blogs = ARTIFACTS_DIR / "blogs"
    if blogs.is_dir():
        for blog in sorted(blogs.iterdir()):
            if blog.is_dir():
                code = blog / "code"
                if code.is_dir():
                    yield code
    # Kernel deep pages
    kernels = ARTIFACTS_DIR / "kernels"
    if kernels.is_dir():
        for slug in sorted(kernels.iterdir()):
            if slug.is_dir():
                for sub in ("full", "variants"):
                    d = slug / sub
                    if d.is_dir():
                        yield d


def find_orphan_source_files():
    """Return list of source files under artifacts/ that are not inside any
    recognized bundle root — these fail AC-2."""
    bundle_roots = set(discover_bundle_roots())
    orphans = []
    if not ARTIFACTS_DIR.is_dir():
        return orphans
    # R33: with `.yaml` now in ASSET_SOURCE_EXTS, per-blog MANIFEST.yaml
    # files at `artifacts/blogs/<slug>/MANIFEST.yaml` would otherwise be
    # flagged as orphans (the recognized bundle root is the `code/`
    # subdir). MANIFEST.yaml is metadata the extractor writes at the
    # parent by design; validate_bundle's drift check already excludes
    # it by name, so mirror that exclusion here. `approach.md` and
    # `bench.txt` are similar bundle-adjacent metadata from earlier
    # rounds that may live above the recognized root.
    _ORPHAN_EXCLUDE_NAMES = {"MANIFEST.yaml", "approach.md", "bench.txt"}
    for path in ARTIFACTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ASSET_SOURCE_EXTS:
            continue
        if path.name in _ORPHAN_EXCLUDE_NAMES:
            continue
        # find nearest bundle root
        in_bundle = any(str(path).startswith(str(root) + "/") or str(path) == str(root) for root in bundle_roots)
        if not in_bundle:
            orphans.append(path)
    return orphans


def validate_bundle(bundle_root, known_source_ids):
    """Validate a single asset bundle root per plan AC-2/AC-9/AC-10."""
    rel = bundle_root.relative_to(REPO_ROOT)
    errors = []
    prov_path = bundle_root / "PROVENANCE.yaml"
    if not prov_path.is_file():
        errors.append(f"{rel}: asset bundle missing PROVENANCE.yaml")
        return errors
    # Disallow nested PROVENANCE.yaml anywhere beneath
    for extra in bundle_root.rglob("PROVENANCE.yaml"):
        if extra != prov_path:
            errors.append(f"{extra.relative_to(REPO_ROOT)}: nested PROVENANCE.yaml disallowed (flat ownership)")
    # Load + schema check
    try:
        prov = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        errors.append(f"{rel}/PROVENANCE.yaml: YAML parse error: {e}")
        return errors
    if not isinstance(prov, dict):
        errors.append(f"{rel}/PROVENANCE.yaml: top-level must be a mapping")
        return errors
    for req in ("origin_url", "upstream_repo", "license", "retrieved_at", "asset_mode", "files"):
        if req not in prov:
            errors.append(f"{rel}/PROVENANCE.yaml: missing required '{req}'")
    mode = prov.get("asset_mode")
    if mode not in ("verbatim", "extracted", "derived"):
        errors.append(f"{rel}/PROVENANCE.yaml: asset_mode '{mode}' must be one of verbatim/extracted/derived")
    if mode == "derived":
        dfrom = prov.get("derived_from")
        if not dfrom or not isinstance(dfrom, list):
            errors.append(f"{rel}/PROVENANCE.yaml: asset_mode=derived requires derived_from list")
        else:
            for sid in dfrom:
                if sid not in known_source_ids:
                    errors.append(f"{rel}/PROVENANCE.yaml: derived_from '{sid}' not a known source id")
    elif mode == "verbatim":
        if not prov.get("upstream_sha"):
            errors.append(f"{rel}/PROVENANCE.yaml: asset_mode=verbatim requires upstream_sha")

    # AC-9 directory-to-mode rule
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "artifacts":
        # Variants dir allows derived; prs/contests disallow derived
        in_variants = len(parts) >= 3 and parts[1] == "kernels" and parts[-1] == "variants"
        if mode == "derived" and not in_variants:
            errors.append(f"{rel}: asset_mode=derived only allowed under artifacts/kernels/*/variants/")
        if not in_variants and parts[1] in ("prs", "contests") and mode not in ("verbatim", "extracted"):
            errors.append(f"{rel}: bundles under artifacts/{parts[1]}/** must use asset_mode verbatim or extracted")

    # Files list validation
    files = prov.get("files") or []
    if not isinstance(files, list):
        errors.append(f"{rel}/PROVENANCE.yaml: files must be a list")
        files = []

    declared_paths = set()
    bundle_total = 0
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] must be a mapping")
            continue
        lp = entry.get("local_path")
        role = entry.get("role")
        e_mode = entry.get("mode")
        sha = entry.get("sha256")
        if not lp:
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] missing local_path")
            continue
        if role not in ("pr-diff", "upstream-file", "extracted-block", "derived-source", "approach-notes", "bench-record"):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].role '{role}' not in allowed set")
        if e_mode not in ("verbatim", "extracted", "derived", "upstream-patch"):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].mode '{e_mode}' not in allowed set")
        if not sha:
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] missing sha256")
        abs_path = bundle_root / lp
        if not abs_path.is_file():
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].local_path '{lp}' does not exist in bundle")
            continue
        # R34: reject manifest entries whose resolved path escapes the
        # bundle root (e.g. `../outside.py`). Without this check,
        # files[*].local_path can satisfy is_file() while pointing at
        # content that isn't actually part of the bundle, undermining
        # the flat-ownership / manifest-drift invariants.
        try:
            resolved_path = abs_path.resolve()
            resolved_root = bundle_root.resolve()
            if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
                errors.append(
                    f"{rel}/PROVENANCE.yaml: files[{i}].local_path '{lp}' escapes "
                    f"the bundle root (resolves to '{resolved_path}', outside '{resolved_root}')"
                )
                continue
        except (OSError, RuntimeError) as e:
            errors.append(
                f"{rel}/PROVENANCE.yaml: files[{i}].local_path '{lp}' could not be resolved: {e}"
            )
            continue
        declared_paths.add(resolved_path)
        # SHA verification (unless size_cap_truncated: true on this entry)
        truncated = bool(entry.get("size_cap_truncated"))
        if sha and not truncated:
            actual = sha256_of_file(abs_path)
            if actual != sha:
                errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] sha256 mismatch (got {actual[:12]}..., declared {str(sha)[:12]}...)")
        # Size cap
        size = abs_path.stat().st_size
        bundle_total += size
        if size > FILE_SIZE_CAP_BYTES and not truncated:
            errors.append(
                f"{rel}/PROVENANCE.yaml: files[{i}] local_path '{lp}' is {size} bytes "
                f"(> {FILE_SIZE_CAP_BYTES} cap); set size_cap_truncated: true or split"
            )
        # Extracted requires heading_path
        if e_mode == "extracted" and not entry.get("heading_path"):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].mode=extracted requires heading_path")

    # Bundle-level size cap
    bundle_truncated = bool(prov.get("size_cap_truncated"))
    if bundle_total > BUNDLE_SIZE_CAP_BYTES and not bundle_truncated:
        errors.append(
            f"{rel}: bundle aggregate is {bundle_total} bytes (> {BUNDLE_SIZE_CAP_BYTES} cap); "
            f"set PROVENANCE.yaml size_cap_truncated: true or downgrade the bundle"
        )

    # Filesystem-vs-manifest drift detection: every source file in the bundle
    # (recursive) must appear in declared_paths
    for f in bundle_root.rglob("*"):
        if not f.is_file():
            continue
        if f.name == "PROVENANCE.yaml":
            continue
        if f.suffix.lower() not in ASSET_SOURCE_EXTS and f.name not in ("MANIFEST.yaml", "approach.md", "bench.txt"):
            continue
        # MANIFEST.yaml (blog extraction) lives in parent, not bundle root
        if f.resolve() not in declared_paths and f.suffix.lower() in ASSET_SOURCE_EXTS:
            errors.append(
                f"{f.relative_to(REPO_ROOT)}: source file present in bundle but not listed in "
                f"{rel}/PROVENANCE.yaml files[*] (manifest drift)"
            )

    return errors


def main():
    tags = load_yaml_file(DATA_DIR / "tags.yaml")
    schemas = load_yaml_file(DATA_DIR / "schemas.yaml")

    all_errors = []
    file_count = 0
    ids_seen = {}

    code_langs = _load_code_langs()

    # First pass: collect all source IDs (for cross-referencing wiki->source).
    # Also collect all known page IDs (source + wiki) for provenance
    # derived_from checks (wiki-hardware/wiki-technique IDs like hw-tcgen05-mma
    # and technique-warp-specialization are legitimate provenance citations).
    all_source_ids = set()
    for md_file in sorted(SOURCES_DIR.rglob("*.md")) if SOURCES_DIR.exists() else []:
        fm = extract_frontmatter(md_file)
        if fm and isinstance(fm, dict) and "id" in fm:
            all_source_ids.add(fm["id"])
    note_ids, note_msg = note_source_ids()
    if note_msg:
        print(f"note: {note_msg}")
    all_source_ids |= note_ids
    all_known_ids = set(all_source_ids)
    for md_file in sorted(WIKI_DIR.rglob("*.md")) if WIKI_DIR.exists() else []:
        fm = extract_frontmatter(md_file)
        if fm and isinstance(fm, dict) and "id" in fm:
            all_known_ids.add(fm["id"])
    global KNOWN_PAGE_IDS, MACHINE_TAGS
    KNOWN_PAGE_IDS = set(all_known_ids)
    MACHINE_TAGS, tag_msg = load_machine_tags()
    if tag_msg:
        print(f"note: {tag_msg}")

    # Second pass: validate everything
    for search_dir in [SOURCES_DIR, WIKI_DIR]:
        if not search_dir.exists():
            continue
        for md_file in sorted(search_dir.rglob("*.md")):
            file_count += 1
            fm = extract_frontmatter(md_file)

            # Check for duplicate ids
            if fm and isinstance(fm, dict) and "id" in fm:
                fid = fm["id"]
                if fid in ids_seen:
                    all_errors.append(
                        f"{md_file.relative_to(REPO_ROOT)}: duplicate id '{fid}' "
                        f"(also in {ids_seen[fid]})"
                    )
                else:
                    ids_seen[fid] = str(md_file.relative_to(REPO_ROOT))

            errors = validate_file(md_file, schemas, tags, all_source_ids, code_langs)
            all_errors.extend(errors)

    # Phase 3: artifact bundle validation
    bundle_count = 0
    bundle_errors = 0
    verbatim_count = 0
    extracted_count = 0
    derived_count = 0
    for bundle_root in discover_bundle_roots():
        bundle_count += 1
        berrs = validate_bundle(bundle_root, all_known_ids)
        if berrs:
            bundle_errors += 1
        all_errors.extend(berrs)
        # Collect asset-mode breakdown for summary
        prov_path = bundle_root / "PROVENANCE.yaml"
        if prov_path.is_file():
            try:
                prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
                m = prov.get("asset_mode")
                if m == "verbatim":
                    verbatim_count += 1
                elif m == "extracted":
                    extracted_count += 1
                elif m == "derived":
                    derived_count += 1
            except yaml.YAMLError:
                pass

    # Template headers cite pages and machine constants in brackets.
    all_errors.extend(template_header_citation_errors())

    # Orphan source-file scan
    orphans = find_orphan_source_files()
    for op in orphans:
        all_errors.append(f"{op.relative_to(REPO_ROOT)}: source file outside any recognized asset bundle")

    # AC-3: candidate-ledger shape check.
    ledger_count = 0
    if CANDIDATES_DIR.exists():
        for ledger_file in sorted(CANDIDATES_DIR.glob("*.yaml")):
            ledger_count += 1
            all_errors.extend(validate_ledger(ledger_file))

    # AC-2 hybrid version-claim registry consistency.
    all_errors.extend(validate_version_claims_registry(all_source_ids))

    # Canonical free-text aliases must agree with the product mapping used by
    # exact architecture filters and cannot collapse a family name to one SM.
    all_errors.extend(validate_alias_contract())

    print(f"Validated {file_count} files ({len(all_source_ids)} source IDs collected)")
    if bundle_count or orphans:
        print(f"Validated {bundle_count} asset bundles "
              f"(verbatim={verbatim_count}, extracted={extracted_count}, derived={derived_count}, "
              f"orphan-source-files={len(orphans)})")
    if ledger_count:
        print(f"Validated {ledger_count} candidate ledgers")
    if all_errors:
        print(f"\n{len(all_errors)} errors found:\n")
        for err in all_errors:
            print(f"  ERROR: {err}")
        sys.exit(1)
    else:
        print("All files valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
