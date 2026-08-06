from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_readme_has_research_only_disclaimer() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "not a medical device" in readme
    assert "does not provide diagnosis" in readme


def test_legacy_archive_has_non_diagnostic_warning() -> None:
    legacy_readme = (REPOSITORY_ROOT / "legacy" / "college-v1" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "historical traceability" in legacy_readme
    assert "must not be interpreted as medical diagnoses" in legacy_readme

