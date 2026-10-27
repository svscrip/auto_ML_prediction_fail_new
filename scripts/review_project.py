"""Reproduce checks or export a bounded, self-contained package for another reviewer.

No upload, checkout, push, package installation or production deployment is performed.
Generated files stay in the ignored artifacts/review directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "41b06a6431a4bf8db9b21590e81f93269ebde71d"
ROOT_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "model_card.md",
    "Model_BPMN.md",
    "Model_retrain_BPMN.md",
    "pyproject.toml",
    "poetry.lock",
    "poetry.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "conftest.py",
    "Dockerfile",
    "Dockerfile.mlflow",
    "docker-compose.yml",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".dockerignore",
    ".env.example",
    ".flake8",
    ".pre-commit-config.yaml",
)


def git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, encoding="utf-8", check=False
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def source_files() -> list[Path]:
    """Explicit allowlist: never include credentials, environments or local runs."""
    files = [ROOT / name for name in ROOT_FILES]
    for folder, suffixes in {
        "src": {".py"},
        "tests": {".py"},
        "scripts": {".py", ".ps1"},
        "docs": {".md", ".yaml", ".bpmn", ".svg", ".png"},
        ".github": {".yml", ".yaml"},
    }.items():
        files.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in suffixes)
    files.extend(ROOT / "keis7-main" / name for name in ("train.csv", "test.csv"))
    files.extend((ROOT / "artifacts").glob("example_*.json"))
    files.extend(
        [
            ROOT / "reference-material/README.md",
            ROOT / "reference-material/keis7-research/model_metadata.json",
        ]
    )
    return sorted({p for p in files if p.is_file()})


def manifest(files: list[Path]) -> dict:
    return {
        p.relative_to(ROOT).as_posix(): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in files
    }


def fresh_directory(kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "artifacts/review" / f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def inspect_predictions(output: Path) -> dict:
    with (ROOT / "keis7-main/test.csv").open(encoding="utf-8", newline="") as file:
        expected_ids = {row["id"] for row in csv.DictReader(file)}
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["id"] for row in rows]
    if len(ids) != len(expected_ids) or set(ids) != expected_ids:
        raise ValueError("Prediction IDs do not match test.csv exactly")
    if not all(
        math.isfinite(float(row["failure_probability"]))
        and 0 <= float(row["failure_probability"]) <= 1
        for row in rows
    ):
        raise ValueError("Invalid prediction probabilities")
    if any(row["risk_level"] not in {"Низкий", "Средний", "Высокий"} for row in rows):
        raise ValueError("Invalid risk label")

    def reject_constant(value: str):
        raise ValueError(f"Non-standard JSON constant: {value}")

    report = json.loads(
        (output / "inference_monitoring.json").read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if report["predictions_rows"] != len(rows):
        raise ValueError("Monitoring row count mismatch")
    return {"rows": len(rows), "ids_match": True, "strict_json": True}


def verify(poetry: str) -> int:
    output = fresh_directory("verification")
    runtime = output / "runtime"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "ML_DATA_DIR": str(ROOT / "keis7-main"),
            "ML_ARTIFACTS_DIR": str(runtime),
            "MLFLOW_TRACKING_URI": f"sqlite:///{(runtime / 'mlflow.db').as_posix()}",
            "MLFLOW_ARTIFACTS_URI": (runtime / "mlartifacts").as_uri(),
            "MLFLOW_EXPERIMENT_NAME": "independent_review",
        }
    )
    source_hashes = manifest(source_files())
    checks = []
    commands = [
        ("poetry-lock", [poetry, "check", "--lock"]),
        (
            "black",
            [
                sys.executable,
                "-m",
                "black",
                "--check",
                "src",
                "tests",
                "scripts",
                "conftest.py",
            ],
        ),
        (
            "isort",
            [
                sys.executable,
                "-m",
                "isort",
                "--check-only",
                "src",
                "tests",
                "scripts",
                "conftest.py",
            ],
        ),
        ("flake8", [sys.executable, "-m", "flake8"]),
        (
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--override-ini",
                "addopts=",
                f"--junitxml={output / 'junit.xml'}",
            ],
        ),
        (
            "model-card",
            [
                sys.executable,
                "-m",
                "src.model_card",
                "--check",
                "model_card.md",
                "--metrics",
                "artifacts/example_model_card_metrics.json",
            ],
        ),
        (
            "train-smoke",
            [sys.executable, "-m", "src.train", "--sample-size", "2000"],
        ),
        (
            "inference",
            [
                sys.executable,
                "-m",
                "src.predict",
                "--model",
                str(runtime / "model.cbm"),
                "--output-dir",
                str(output / "inference"),
            ],
        ),
    ]
    for name, command in commands:
        failed_prerequisites = [
            c["name"]
            for c in checks
            if c["name"] in ("pytest", "train-smoke") and c["returncode"] != 0
        ]
        if name in ("train-smoke", "inference") and failed_prerequisites:
            reason = f"{', '.join(failed_prerequisites)} failed"
            checks.append({"name": name, "returncode": None, "reason": reason})
            continue
        log = output / f"{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    check=False,
                    timeout=900,
                )
                code = result.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                stream.write(f"{type(exc).__name__}: {exc}\n")
                code = -1
        checks.append(
            {"name": name, "command": command, "returncode": code, "log": log.name}
        )
        print(f"{name}: {'PASS' if code == 0 else 'FAIL'}", flush=True)
    try:
        details = inspect_predictions(output / "inference")
        checks.append(
            {"name": "prediction-contract", "returncode": 0, "details": details}
        )
    except (OSError, ValueError, KeyError) as exc:
        checks.append(
            {"name": "prediction-contract", "returncode": 1, "error": str(exc)}
        )
    unchanged = source_hashes == manifest(source_files())
    passed = all(check["returncode"] == 0 for check in checks) and unchanged
    write_json(
        output / "verification.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "head_sha": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "working_tree_status": git("status", "--porcelain"),
            "base_sha": BASE_SHA,
            "python": sys.version,
            "platform": platform.platform(),
            "all_passed": passed,
            "source_files_unchanged": unchanged,
            "checks": checks,
            "source_manifest": source_hashes,
            "limits": [
                "Smoke training, not full training",
                "No Docker run by this script",
                "No production deployment",
                "No external AI review performed",
            ],
        },
    )
    print(f"Report: {output / 'verification.json'}", flush=True)
    return 0 if passed else 1


def bundle(verification: Path) -> int:
    verification = verification.resolve()
    report = json.loads(
        (verification / "verification.json").read_text(encoding="utf-8")
    )
    files = source_files()
    current = manifest(files)
    if report["source_manifest"] != current:
        raise ValueError("Sources changed since verification. Run verify again.")
    output = fresh_directory("packet")
    metadata = {
        "head_sha": git("rev-parse", "HEAD"),
        "base_sha": BASE_SHA,
        "verified_head_sha": report["head_sha"],
        "working_tree_status": git("status", "--porcelain"),
        "source_manifest": current,
        "all_checks_passed": report["all_passed"],
        "omissions": [
            ".git",
            ".venv",
            "secrets",
            "training outputs",
            "lecture recordings/slides",
            "research notebooks and binaries",
        ],
    }
    write_json(output / "manifest.json", metadata)
    prompt = (ROOT / "docs/AI_REVIEW.md").read_text(encoding="utf-8")
    chunks = [
        prompt,
        "\n\n===== SNAPSHOT MANIFEST =====\n",
        json.dumps(metadata, ensure_ascii=False, indent=2),
    ]
    text_files = [
        p
        for p in files
        if (
            p.suffix
            in {".py", ".ps1", ".md", ".toml", ".yaml", ".yml", ".json", ".bpmn"}
            or p.name
            in {
                "Dockerfile",
                "Dockerfile.mlflow",
                ".gitignore",
                ".gitattributes",
                ".editorconfig",
                ".dockerignore",
                ".env.example",
                ".flake8",
                "requirements.txt",
            }
        )
        and p.name != "AI_REVIEW.md"
    ]
    for path in text_files:
        chunks.extend(
            [
                f"\n\n===== FILE: {path.relative_to(ROOT).as_posix()} =====\n",
                path.read_text(encoding="utf-8-sig"),
            ]
        )
    chunks.extend(
        [
            "\n\n===== VERIFICATION RESULT =====\n",
            json.dumps(report, ensure_ascii=False, indent=2),
        ]
    )
    (output / "AI_REVIEW_PACKET.txt").write_text("".join(chunks), encoding="utf-8")
    diff = git("diff", "--no-ext-diff", BASE_SHA, "--", *current.keys())
    (output / "changes-from-main.patch").write_text(diff + "\n", encoding="utf-8")
    shutil.copyfile(verification / "verification.json", output / "verification.json")
    archive = output / "yanshin-review.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in files:
            zip_file.write(path, "project/" + path.relative_to(ROOT).as_posix())
        for path in output.iterdir():
            if path != archive and path.is_file():
                zip_file.write(path, path.name)
        for path in verification.iterdir():
            if path.suffix in {".log", ".xml"}:
                zip_file.write(path, "evidence/" + path.name)
    print(f"Text for AI: {output / 'AI_REVIEW_PACKET.txt'}")
    print(f"Archive: {archive}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    check = subparsers.add_parser(
        "verify", help="Run checks in a separate local artifact store"
    )
    check.add_argument("--poetry", default="poetry", help="Poetry executable path")
    pack = subparsers.add_parser(
        "bundle", help="Package the exact verified sources for a reviewer"
    )
    pack.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    return verify(args.poetry) if args.action == "verify" else bundle(args.verification)


if __name__ == "__main__":
    raise SystemExit(main())
