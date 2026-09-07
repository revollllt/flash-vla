"""Contract tests for validate.py: the rules this wiki adds to KernelWiki's.

Run: .venv/bin/python -m unittest discover -s .claude/skills/kernel-wiki/tests
"""
import importlib.util
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("validate_contracts", SCRIPTS / "validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)

TAGS = {
    "hardware_features": ["wgmma", "tma"],
    "techniques": ["pipeline-stages"],
    "kernel_types": ["gemm"],
    "languages": ["cuda-cpp"],
    "architectures": ["sm90"],
    "confidence": ["verified", "source-reported", "inferred", "experimental", "measured"],
    "reproducibility": ["concept", "pseudocode", "snippet", "runnable", "benchmarked"],
    "symptoms": ["stall-gmma", "memory-bound"],
    "source_categories": ["official-doc"],
}
SCHEMAS = {
    "wiki-technique": {"required": ["id", "title", "type", "sources"], "constraints": {"type": "technique", "id_prefix": "technique-", "reproducibility_minimum": "snippet"}},
    "wiki-pattern": {"required": ["id", "title", "type", "symptoms", "candidate_techniques", "sources"], "constraints": {"type": "pattern", "id_prefix": "pattern-"}},
}


@contextmanager
def validation_root(root):
    names = ("REPO_ROOT", "SOURCES_DIR", "WIKI_DIR", "DATA_DIR", "ARTIFACTS_DIR", "CANDIDATES_DIR", "KNOWN_PAGE_IDS", "MACHINE_TAGS")
    previous = {name: getattr(validate, name) for name in names}
    validate.REPO_ROOT = root
    validate.SOURCES_DIR = root / "sources"
    validate.WIKI_DIR = root / "wiki"
    validate.DATA_DIR = root / "data"
    validate.ARTIFACTS_DIR = root / "artifacts"
    validate.CANDIDATES_DIR = root / "candidates"
    validate.KNOWN_PAGE_IDS = {"technique-a", "hw-tma"}
    validate.MACHINE_TAGS = {"wgmma.issue.wg.ss"}
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(validate, name, value)


def run_page(subdir, text):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        page = root / "wiki" / subdir / "example.md"
        page.parent.mkdir(parents=True)
        page.write_text(text, encoding="utf-8")
        with validation_root(root):
            return validate.validate_file(page, SCHEMAS, TAGS, {"doc-x", "note-y"}, {"cpp", "cuda"})


CODE = "\n```cpp\nint main() {\n  return 0;\n}\n```\n"


class MeasuredConfidenceTests(unittest.TestCase):
    def page(self, extra):
        return ("---\nid: technique-example\ntitle: t\ntype: technique\ntags: [wgmma]\n"
                "confidence: measured\nreproducibility: snippet\nsources: [doc-x, note-y]\n"
                f"{extra}---\n\n# T\n{CODE}")

    def test_measured_requires_benchmark_or_reproduction_evidence(self):
        errors = run_page("techniques", self.page(""))
        self.assertTrue(any("'measured' requires" in e for e in errors), errors)

    def test_measured_with_benchmark_evidence_passes(self):
        errors = run_page("techniques", self.page("evidence_basis:\n  - evidence_type: benchmark\n    source_id: note-y\n"))
        self.assertEqual([e for e in errors if "measured" in e], [])

    def test_evidence_source_must_be_in_page_sources(self):
        errors = run_page("techniques", self.page("evidence_basis:\n  - evidence_type: benchmark\n    source_id: doc-z\n"))
        self.assertTrue(any("not listed in page sources" in e for e in errors), errors)


class VocabularyAndReferenceTests(unittest.TestCase):
    def pattern(self, symptoms, candidates, body=""):
        return ("---\nid: pattern-example\ntitle: t\ntype: pattern\ntags: [wgmma]\n"
                f"symptoms: [{symptoms}]\ncandidate_techniques: [{candidates}]\nsources: [doc-x]\n---\n\n# P\n{body}")

    def test_unknown_symptom_is_rejected(self):
        errors = run_page("patterns", self.pattern("not-a-symptom", "technique-a"))
        self.assertTrue(any("not a valid symptoms value" in e for e in errors), errors)

    def test_dangling_candidate_is_rejected(self):
        errors = run_page("patterns", self.pattern("stall-gmma", "technique-missing"))
        self.assertTrue(any("does not resolve to a page id" in e for e in errors), errors)

    def test_machine_tag_must_resolve(self):
        errors = run_page("patterns", self.pattern("stall-gmma", "technique-a", "Costs [wgmma.issue.wg.ss] and [no.such.tag].\n"))
        self.assertTrue(any("[no.such.tag]" in e for e in errors), errors)
        self.assertFalse(any("[wgmma.issue.wg.ss]" in e for e in errors), errors)

    def test_relative_link_must_resolve(self):
        errors = run_page("patterns", self.pattern("stall-gmma", "technique-a", "See [x](../techniques/missing.md).\n"))
        self.assertTrue(any("relative link" in e for e in errors), errors)

    def test_excerpt_must_be_verbatim_in_named_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "artifacts" / "kernels" / "sm90-templates" / "variants"
            bundle.mkdir(parents=True)
            (bundle / "99_example.cu").write_text("int a = 1;\nint b = 2;\nint c = 3;\n", encoding="utf-8")
            prev = validate.TEMPLATE_BUNDLE
            validate.TEMPLATE_BUNDLE = bundle
            try:
                ok = validate.excerpt_errors("From `99_example.cu`:\n\n```cpp\nint b = 2;\nint c = 3;\n```\n", "p")
                bad = validate.excerpt_errors("From `99_example.cu`:\n\n```cpp\nint b = 2;\nint z = 9;\n```\n", "p")
            finally:
                validate.TEMPLATE_BUNDLE = prev
        self.assertEqual(ok, [])
        self.assertTrue(bad and "not found verbatim" in bad[0], bad)

    def test_note_source_id_shape(self):
        ids, note = validate.note_source_ids()
        if note is None:
            self.assertTrue(all(i.startswith("note-") for i in ids))


if __name__ == "__main__":
    unittest.main()
