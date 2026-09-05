#!/usr/bin/env python3
"""Check the sm90 wiki against the rules its own README states.

The wiki's value is that a reader can trust its shape: front matter that says
what kind of entry this is and how much it is believed, a symptom row that
routes to it, citations that resolve, and no project history leaking in.  Every
one of those decays silently -- a renamed constant, a deleted entry still linked
from three others, an entry nobody can reach from the routing table.  This
script fails on each of them so the decay is a build error rather than a reader
discovering it years later.

    python3 check_wiki.py                 # every check
    python3 check_wiki.py --only pdl      # entries whose name contains "pdl"
    python3 check_wiki.py --no-tags       # skip constants.py resolution

    Exit codes: 0 clean, 1 a check failed, 2 the environment cannot run it.

What it does NOT prove: that an entry is true, that its rule still holds, or
that its confidence level is honest.  Those come from the measurements behind
it, which live in Agent Notes and per-task workspaces, never here.
"""

import argparse
import os
import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
WIKI_DIR = SKILL_DIR / "references" / "wiki"
TEMPLATE_DIR = SKILL_DIR / "references" / "templates"
# .claude/skills/kernel-design -> .claude/skills
SKILLS_DIR = SKILL_DIR.parent
CONSTANTS_PY = SKILLS_DIR / "hardware-unit-test" / "scripts" / "constants.py"

# The entry vocabulary, from wiki/README.md's "Entry format".  Kept here rather
# than parsed out of the README because a typo in the README would otherwise
# widen the vocabulary instead of failing.
FRONT_MATTER_KEYS = ["id", "type", "arch", "tags", "confidence"]
TYPES = {"pattern", "technique", "case", "external"}
CONFIDENCE = {"measured", "source-reported", "inferred"}

# Section order per type.  External entries are a different genre -- they
# distil somebody else's work, so they open with what to take from it and close
# with where it came from -- and the corpus has always spelled them that way.
REQUIRED_SECTIONS = {
    "external": ["What to read it for", "Source"],
    "*": ["Context", "Move"],
}

# The README's rule "entries are distilled experience, never experiment
# records", made mechanical.  Narrow on purpose: each pattern is something that
# cannot appear in portable experience, so a hit is a defect and not a style
# opinion.
FORBIDDEN = [
    (re.compile(r"\bSLURM[ _]?\d{4,}", re.I), "cites a job id"),
    (re.compile(r"\bjob \d{4,}\b", re.I), "cites a job id"),
    (re.compile(r"\b(?:src|eval|benchmarks|artifacts)/[a-z_]+/"), "cites a project path"),
    (re.compile(r"\bflash_vla\b"), "names the project"),
    (re.compile(r"\bour (?:kernel|pipeline|target|repo)\b", re.I), "narrates this project"),
]

FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
FENCE_RE = re.compile(r"```.*?```", re.S)
MD_LINK_RE = re.compile(r"\[([^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
BRACKET_RE = re.compile(r"\[([^\]\n]{1,120})\]")
# `<engine>.<quantity>.<scope>[.<condition>]` -- hardware-unit-test's grammar.
TAG_RE = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9]+){2,3}$")
# An entry id: lowercase kebab, at least two words.
ENTRY_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")


def entries():
    return sorted(p for p in WIKI_DIR.glob("*.md") if p.stem != "README")


def prose(path):
    """The text a citation can legitimately appear in.

    Front matter is stripped because `tags: [pdl, fusion]` is a taxonomy, not a
    citation; fenced blocks because a format example is not a claim; markdown
    link syntax because `[text](target)` is checked as a link instead.  In a
    template only comment lines survive, so an array subscript is never read as
    a citation.
    """
    if path.suffix in (".cu", ".cuh"):
        return "\n".join(l for l in path.read_text().splitlines()
                         if l.lstrip().startswith("//"))
    text = FRONT_MATTER_RE.sub("", path.read_text())
    text = FENCE_RE.sub("", text)
    return MD_LINK_RE.sub(lambda m: m.group(1), text)


def load_front_matter(path):
    """Returns (mapping, failures).  A missing block is one failure, not five."""
    import yaml

    match = FRONT_MATTER_RE.match(path.read_text())
    if not match:
        return None, ["no `---` front matter block at the top"]
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return None, [f"front matter is not valid YAML: {exc}"]
    if not isinstance(data, dict):
        return None, "front matter is not a mapping"
    return data, []


def check_front_matter(path, data):
    out = []
    keys = list(data.keys())
    missing = [k for k in FRONT_MATTER_KEYS if k not in keys]
    extra = [k for k in keys if k not in FRONT_MATTER_KEYS]
    if missing:
        out.append(f"front matter missing {missing}")
    if extra:
        out.append(f"front matter has undeclared keys {extra}")
    if keys[:len(FRONT_MATTER_KEYS)] != FRONT_MATTER_KEYS and not missing and not extra:
        out.append(f"front matter key order is {keys}, expected {FRONT_MATTER_KEYS}")
    if data.get("id") != path.stem:
        out.append(f"id is {data.get('id')!r}, must equal the file stem {path.stem!r}")
    if data.get("type") not in TYPES:
        out.append(f"type is {data.get('type')!r}, not one of {sorted(TYPES)}")
    if data.get("confidence") not in CONFIDENCE:
        out.append(f"confidence is {data.get('confidence')!r}, not one of {sorted(CONFIDENCE)}")
    if not data.get("arch"):
        out.append("arch is empty")
    tags = data.get("tags")
    if not isinstance(tags, list) or not tags:
        out.append("tags must be a non-empty list")
    return out


def check_structure(path, data):
    """One H1, and the sections the entry's type requires, in order."""
    body = FRONT_MATTER_RE.sub("", path.read_text())
    h1 = re.findall(r"^# (.+)$", body, re.M)
    out = []
    if len(h1) != 1:
        out.append(f"has {len(h1)} `# ` titles, expected exactly 1")
    headings = re.findall(r"^## (.+)$", body, re.M)
    required = REQUIRED_SECTIONS.get(data.get("type"), REQUIRED_SECTIONS["*"])
    # Section titles may carry a trailing qualifier ("Caveats -- two traps"),
    # so match on the prefix rather than on equality.
    for i, want in enumerate(required):
        if not any(h.startswith(want) for h in headings):
            out.append(f"missing a `## {want}` section")
        elif i and required[i - 1] in [h.split(" --")[0] for h in headings]:
            first = next(j for j, h in enumerate(headings) if h.startswith(required[i - 1]))
            here = next(j for j, h in enumerate(headings) if h.startswith(want))
            if want == "Source":
                continue  # Source closes the entry; its position is not ordered against Context
            if here < first:
                out.append(f"`## {want}` appears before `## {required[i - 1]}`")
    return out


def check_hygiene(path):
    out = []
    for line_no, line in enumerate(prose(path).splitlines(), 1):
        for pattern, why in FORBIDDEN:
            if pattern.search(line):
                out.append(f"line ~{line_no} {why}: {line.strip()[:70]}")
    return out


def check_routing(readme_text, ids):
    """Every entry needs a symptom row, or nobody arrives at it.

    The routing table is the wiki's only index; an entry absent from it is
    reachable by grep alone, which is the failure mode the table exists to
    prevent.
    """
    table = "\n".join(l for l in readme_text.splitlines() if l.startswith("|"))
    return [f"{i} has no row in the routing table" for i in sorted(ids)
            if i not in table]


def check_links(paths):
    """Relative markdown links must resolve on disk, anchors stripped."""
    out = []
    for path in paths:
        for text, target in MD_LINK_RE.findall(path.read_text()):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                out.append(f"{path.name}: dead link [{text[:40]}]({target})")
    return out


def collect_citations(paths):
    """Bracket citations, classified by shape: {tag: files}, {entry id: files}."""
    tags, refs = {}, {}
    for path in paths:
        for match in BRACKET_RE.finditer(prose(path)):
            for token in (t.strip() for t in match.group(1).split(",")):
                if TAG_RE.match(token):
                    tags.setdefault(token, set()).add(path.name)
                elif ENTRY_ID_RE.match(token):
                    refs.setdefault(token, set()).add(path.name)
    return tags, refs


def known_tags():
    """Ask constants.py, which owns the tag namespace.

    Returns (set, note).  A missing sibling skill is a note rather than a
    failure: this wiki is portable and may travel without hardware-unit-test,
    and a checker that cannot run is worse than one that says what it skipped.
    """
    if not CONSTANTS_PY.exists():
        return None, f"no {CONSTANTS_PY.relative_to(SKILLS_DIR)}; tag resolution skipped"
    sys.path.insert(0, str(CONSTANTS_PY.parent))
    try:
        import constants  # the authority on what a tag means
        return {c["tag"] for _, doc in constants.load() for c in doc.get("constants", [])}, None
    except Exception as exc:  # noqa: BLE001 -- any import/parse failure is the same note
        return None, f"constants.py unusable ({exc}); tag resolution skipped"
    finally:
        sys.path.pop(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="substring filter on entry file name")
    ap.add_argument("--no-tags", action="store_true",
                    help="skip machine-constant tag resolution")
    args = ap.parse_args()

    try:
        import yaml  # noqa: F401
    except ImportError:
        print("check_wiki needs PyYAML (the same dependency constants.py has)",
              file=sys.stderr)
        return 2

    all_entries = entries()
    if not all_entries:
        print(f"no entries under {WIKI_DIR}", file=sys.stderr)
        return 1
    ids = {p.stem for p in all_entries}
    selected = [p for p in all_entries if not args.only or args.only in p.name]

    failed = 0
    for path in selected:
        data, notes = load_front_matter(path)
        if data is not None:
            notes = (check_front_matter(path, data) + check_structure(path, data)
                     + check_hygiene(path))
        print(f"{'PASS' if not notes else 'FAIL'}  {path.name}")
        for note in notes:
            print(f"      {note}")
        failed += bool(notes)

    print()
    corpus = []

    readme = WIKI_DIR / "README.md"
    corpus += [("routing", n) for n in check_routing(readme.read_text(), ids)]

    linked = all_entries + [readme, TEMPLATE_DIR / "README.md"]
    corpus += [("links", n) for n in check_links(linked)]

    cited = all_entries + [readme, TEMPLATE_DIR / "README.md"]
    cited += sorted(TEMPLATE_DIR.glob("*.cu")) + sorted(TEMPLATE_DIR.glob("*.cuh"))
    tags, refs = collect_citations(cited)
    for ref, where in sorted(refs.items()):
        if ref not in ids:
            corpus.append(("citations", f"[{ref}] in {sorted(where)} is not a wiki entry"))

    tag_note = None
    if args.no_tags:
        tag_note = "tag resolution skipped (--no-tags)"
    else:
        available, tag_note = known_tags()
        if available is not None:
            for tag, where in sorted(tags.items()):
                if tag not in available:
                    corpus.append(("citations",
                                   f"[{tag}] in {sorted(where)} resolves to no constant"))

    for kind, note in corpus:
        print(f"FAIL  {kind:10s} {note}")
    if tag_note:
        print(f"note: {tag_note}")

    print(f"\n{len(selected) - failed}/{len(selected)} entries pass; "
          f"{len(corpus)} corpus-wide failures "
          f"({len(tags)} tag citations, {len(refs)} entry citations checked)")
    return 1 if failed or corpus else 0


if __name__ == "__main__":
    sys.exit(main())
