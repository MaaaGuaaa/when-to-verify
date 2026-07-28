"""Repository guards for contracts that remain retired after SOP08."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

RETIRED_PATHS = (
    "src/evaluation/verification_baselines.py",
    "src/evaluation/verification_metrics.py",
    "src/models/verification_model.py",
    "src/models/verification_training.py",
    "src/datasets/snippet_library.py",
    "src/datasets/risk_dataset.py",
    "src/datasets/sop03_publication.py",
    "src/datasets/verification_collection.py",
    "src/datasets/verification_dataloader.py",
    "src/datasets/verification_dataset.py",
    "src/datasets/verification_sources.py",
    "src/generation/dynamic_object_transplant.py",
    "src/generation/event_sampler.py",
    "src/generation/paired_variants.py",
    "src/generation/sop05_input_adapter.py",
    "src/generation/sop05_output_loader.py",
    "src/generation/sop05_run.py",
    "src/generation/sop05r_event_sampler.py",
    "src/generation/sop05r_output_loader.py",
    "src/generation/sop05r_revealability.py",
    "src/generation/sop05r_run.py",
    "src/generation/sop05r_trajectory_store.py",
    "src/generation/sop06_pipeline.py",
    "scripts/04_generate_risk_dataset.py",
    "scripts/08_generate_verification_dataset.py",
    "scripts/09_train_verification_model.py",
    "scripts/10_evaluate_verification_model.py",
)

RETIRED_CONTRACT_SYMBOLS = (
    "VerificationSample",
    "validate_verification_sample",
)

CURRENT_BASE_CONFIG_KEYS = {
    "seed",
    "schema_version",
    "bev",
    "robot",
    "dynamic_objects",
}

FORBIDDEN_PRODUCTION_PATTERNS = (
    re.compile(r"history8_current7_future15"),
    re.compile(r"future_steps\s*[=:]\s*15\b"),
    re.compile(r"[\"']future_steps[\"']\s*:\s*15\b"),
    re.compile(r"trajectory_steps[^\n\d]*15\b"),
    re.compile(r"sample_count[^\n\d]*23\b"),
    re.compile(r"MOTION_SNIPPET_(?:FUTURE_STEPS|SAMPLE_COUNT)\s*=\s*(?:15|23)\b"),
    re.compile(r"\(15,\s*3\)"),
)


def test_retired_paths_are_absent() -> None:
    remaining = [path for path in RETIRED_PATHS if (ROOT / path).exists()]
    assert remaining == []


def test_retired_post_sop08_contract_symbols_are_absent() -> None:
    from src import contracts

    remaining = [
        name for name in RETIRED_CONTRACT_SYMBOLS if hasattr(contracts, name)
    ]
    assert remaining == []


def test_base_config_contains_only_current_long40_sections() -> None:
    from src.utils.config import load_config

    assert set(load_config(ROOT / "configs/base.yaml")) == CURRENT_BASE_CONFIG_KEYS


def test_production_tree_has_no_short_horizon_contract_literals() -> None:
    matches: list[str] = []
    for directory in ("scripts", "src", "configs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if any(pattern.search(line) for pattern in FORBIDDEN_PRODUCTION_PATTERNS):
                    matches.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    assert matches == []
