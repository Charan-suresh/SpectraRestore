#!/usr/bin/env python3
"""
SpectraRestore Submission Validator & Packaging Tool

Validates the full repository state and generates a clean, compliant
submission archive (submission_SpectraRestore.zip).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def check_submission_artifacts(root: Path = ROOT) -> Tuple[bool, Dict[str, bool]]:
    """Verify presence of all mandatory submission files."""
    checklist = {
        "run.py": (root / "run.py").is_file(),
        "requirements.txt": (root / "requirements.txt").is_file(),
        "README.md": (root / "README.md").is_file(),
        "models/*.pt": (
            (len(list((root / "models").glob("*.pt"))) > 0 if (root / "models").is_dir() else False)
            or (len(list((root / "weights").glob("*.pt"))) > 0 if (root / "weights").is_dir() else False)
        ),
        "src/model.py": (root / "src" / "model.py").is_file(),
        "src/train.py": (root / "src" / "train.py").is_file(),
        "evaluate.py": (root / "evaluate.py").is_file(),
    }

    all_passed = all(checklist.values())

    print("=" * 60)
    print("SPECTRARESTORE SUBMISSION CHECK")
    print("=" * 60)
    for item, status in checklist.items():
        tag = "PASS" if status else "MISSING"
        icon = "✅" if status else "❌"
        print(f"{item:<25} {tag:>10}  {icon}")
    print("-" * 60)
    if all_passed:
        print("SUBMISSION CHECK: PASS")
    else:
        print("SUBMISSION CHECK: FAILED (some artifacts are missing)")
    print("=" * 60)

    return all_passed, checklist


def create_submission_zip(
    output_zip: Path, root: Path = ROOT, strict: bool = False
) -> Path:
    """Create a clean submission zip archive excluding datasets, git, and caches."""
    passed, checklist = check_submission_artifacts(root)
    if strict and not passed:
        missing = [k for k, v in checklist.items() if not v]
        raise RuntimeError(f"Cannot package submission: missing artifacts ({', '.join(missing)})")

    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    exclude_patterns = {
        ".git",
        ".github",
        ".idea",
        ".vscode",
        ".DS_Store",
        "__pycache__",
        ".pytest_cache",
        "data",  # exclude raw dataset
        "venv",
        ".venv",
        "env",
    }

    included_count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(root)
            parts = rel.parts

            # Check if any path segment is in exclude list
            if any(part in exclude_patterns for part in parts):
                continue
            if p.name.startswith("._") or p.name.endswith(".pyc"):
                continue
            if p == output_zip:
                continue

            zf.write(p, rel)
            included_count += 1

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\n[package] Created {output_zip.name} ({size_mb:.2f} MB, {included_count} files)")
    return output_zip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpectraRestore Submission Packaging Tool")
    p.add_argument("--out", type=str, default="submission_SpectraRestore.zip", help="Output zip filename")
    p.add_argument("--strict", action="store_true", help="Fail if any required artifact is missing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_zip = Path(args.out)
    try:
        create_submission_zip(out_zip, strict=args.strict)
    except Exception as e:
        print(f"[package] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
