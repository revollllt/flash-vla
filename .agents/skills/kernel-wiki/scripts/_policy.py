"""Architecture vocabulary and the generated source-PR body contract.

Extracted from KernelWiki's ``scripts/pr_policy.py`` (MIT, mit-han-lab): the
constants and two functions that ``validate.py``, ``query.py`` and
``generate-indices.py`` consume. The PR-intake classifier that surrounded
them in the upstream file is not carried here; this wiki does not ingest a
PR corpus, it cites the PR pages it copied.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

BODY_CONTRACT = "upstream-pr-v1"
UPSTREAM_EXCERPT_LIMIT = 1200

ARCHITECTURE_FAMILY_PREFIXES = {
    "turing": ("sm75",),
    "ampere": ("sm80", "sm86", "sm87", "sm88"),
    "ada": ("sm89",),
    "hopper": ("sm90",),
    "blackwell": ("sm100", "sm103", "sm110", "sm120", "sm121"),
}
BLACKWELL_EXACT_PREFIXES = ARCHITECTURE_FAMILY_PREFIXES["blackwell"]

# Canonical exact targets accepted by both extraction and schema validation.
SUPPORTED_EXACT_ARCHITECTURES = (
    "sm75",
    "sm80", "sm86", "sm87", "sm88", "sm89",
    "sm90", "sm90a",
    "sm100", "sm100a", "sm100f",
    "sm103", "sm103a", "sm103f",
    "sm110", "sm110a", "sm110f",
    "sm120", "sm120a", "sm120f",
    "sm121", "sm121a", "sm121f",
)

# Product name -> (exact architecture, documenting URL).
PRODUCT_ARCHITECTURE_MAPPINGS = {
    "b200": ("sm100", "https://developer.nvidia.com/cuda/gpus"),
    "gb200": ("sm100", "https://developer.nvidia.com/cuda/gpus"),
    "b300": ("sm103", "https://developer.nvidia.com/cuda/gpus"),
    "gb300": ("sm103", "https://developer.nvidia.com/cuda/gpus"),
    "h100": ("sm90", "https://developer.nvidia.com/cuda/gpus"),
    "h200": ("sm90", "https://developer.nvidia.com/cuda/gpus"),
    "gh200": ("sm90", "https://developer.nvidia.com/cuda/gpus"),
    "h800": ("sm90", "https://docs.omniverse.nvidia.com/dang/latest/common/technical-requirements.html"),
    "h20": ("sm90", "https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html"),
    "a100": ("sm80", "https://developer.nvidia.com/cuda/gpus"),
}


def upstream_excerpt(body: str) -> str:
    canonical_body = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    canonical_lines = []
    for line in canonical_body.split("\n"):
        line = line.rstrip(" \t")
        if re.match(r"^(?:<{7}|={7}|>{7})(?: |$)", line):
            line = "\\" + line
        canonical_lines.append(line)
    canonical_body = "\n".join(canonical_lines)
    excerpt = canonical_body.strip()[:UPSTREAM_EXCERPT_LIMIT].rstrip()
    return excerpt.replace("<!-- upstream-excerpt-start -->", "&lt;!-- upstream-excerpt-start --&gt;").replace(
        "<!-- upstream-excerpt-end -->", "&lt;!-- upstream-excerpt-end --&gt;"
    )


def render_generated_body(body: str, changed_paths: Iterable[str], total_files: int | None = None) -> tuple[str, str]:
    """Render the only allowed generated source-PR body template."""
    excerpt = upstream_excerpt(body)
    excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    paths = list(changed_paths)
    total = len(paths) if total_files is None else total_files
    lines = [
        "## Upstream PR description (verbatim excerpt)",
        "",
        "<!-- upstream-excerpt-start -->",
        excerpt if excerpt else "_The upstream PR description is empty._",
        "<!-- upstream-excerpt-end -->",
        "",
        "## Changed files (upstream)",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    if total > len(paths):
        lines.append(f"- _{total - len(paths)} additional changed file(s) omitted from this display._")
    lines.append("")
    return "\n".join(lines), excerpt_hash


def body_contract_errors(frontmatter: dict[str, Any], body: str) -> list[str]:
    """Validate the offline generated-body shape and its excerpt digest."""
    errors = []
    if frontmatter.get("body_contract") != BODY_CONTRACT:
        return [f"body_contract must be {BODY_CONTRACT!r}"]
    start = "<!-- upstream-excerpt-start -->\n"
    end = "\n<!-- upstream-excerpt-end -->"
    if body.count(start) != 1 or body.count(end) != 1:
        errors.append("generated body must contain exactly one upstream excerpt sentinel pair")
        return errors
    prefix, rest = body.split(start, 1)
    excerpt_rendered, suffix = rest.split(end, 1)
    if prefix != "## Upstream PR description (verbatim excerpt)\n\n":
        errors.append("generated body has text outside the upstream-pr-v1 prefix template")
    if not suffix.startswith("\n\n## Changed files (upstream)\n\n"):
        errors.append("generated body has text outside the upstream-pr-v1 suffix template")
    empty_template = "_The upstream PR description is empty._"
    excerpt = "" if excerpt_rendered == empty_template else excerpt_rendered
    actual = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    if frontmatter.get("upstream_excerpt_sha256") != actual:
        errors.append("upstream_excerpt_sha256 does not match rendered upstream excerpt")
    if not re.fullmatch(r"[0-9a-f]{64}", str(frontmatter.get("upstream_body_sha256", ""))):
        errors.append("upstream_body_sha256 must be a full SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", str(frontmatter.get("upstream_files_sha256", ""))):
        errors.append("upstream_files_sha256 must be a full SHA-256 digest")
    expected_body, _ = render_generated_body(
        excerpt,
        frontmatter.get("changed_paths") or [],
        frontmatter.get("changed_files_count"),
    )
    if body != expected_body:
        errors.append("generated body differs from the deterministic upstream-pr-v1 template")
    return errors


def architecture_matches_filter(architectures: Iterable[str], disposition: str, requested: str) -> bool:
    """Canonical user-facing architecture hierarchy semantics."""
    archs = {str(value).lower() for value in architectures or []}
    requested = requested.lower().replace("_", "")
    if requested in PRODUCT_ARCHITECTURE_MAPPINGS:
        requested = PRODUCT_ARCHITECTURE_MAPPINGS[requested][0]
    if requested == "unknown":
        return disposition == "unknown" and not archs
    if requested in ARCHITECTURE_FAMILY_PREFIXES:
        return requested in archs or any(
            value.startswith(ARCHITECTURE_FAMILY_PREFIXES[requested]) for value in archs
        )
    return requested in archs
