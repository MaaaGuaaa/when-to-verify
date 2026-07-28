"""Guards against reintroducing the retired short-horizon production contract."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PRODUCTION_PATTERNS = (
    re.compile(r"history8_current7_future15"),
    re.compile(r"future_steps\s*[=:]\s*15\b"),
    re.compile(r"[\"']future_steps[\"']\s*:\s*15\b"),
    re.compile(r"trajectory_steps[^\n\d]*15\b"),
    re.compile(r"sample_count[^\n\d]*23\b"),
    re.compile(r"MOTION_SNIPPET_(?:FUTURE_STEPS|SAMPLE_COUNT)\s*=\s*(?:15|23)\b"),
    re.compile(r"\(15,\s*3\)"),
)

RETIRED_PRODUCTION_PATHS = (
    "configs/generator_seen_occluded_visual_audit.yaml",
    "configs/generator_test.yaml",
    "configs/generator_train.yaml",
    "configs/paired_variants.yaml",
    "configs/paired_variants_visual_audit.yaml",
    "configs/seen_occluded_joint_visual_audit.yaml",
    "scripts/02_build_snippet_library.py",
    "scripts/03_finalize_sop03_artifact.py",
    "scripts/04_build_trajectory_bank.py",
    "scripts/04_generate_risk_dataset.py",
    "scripts/04_publish_risk_evaluation_records.py",
    "scripts/05_render_seen_occluded_visual_audit.py",
    "scripts/05_render_sop05r_audit.py",
    "src/datasets/snippet_library.py",
    "src/datasets/sop03_publication.py",
    "src/datasets/verification_sources.py",
    "src/evaluation/seen_occluded_visual_audit.py",
    "src/evaluation/seen_occluded_visuals.py",
    "src/evaluation/sop05r_audit.py",
    "src/evaluation/sop05r_visuals.py",
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
)


def test_retired_15_step_production_paths_are_absent() -> None:
    remaining = [path for path in RETIRED_PRODUCTION_PATHS if (ROOT / path).exists()]
    assert remaining == []


def test_production_tree_has_no_retired_15_step_contract() -> None:
    matches: list[str] = []
    for directory in ("scripts", "src", "configs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".json"}:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if any(pattern.search(line) for pattern in FORBIDDEN_PRODUCTION_PATTERNS):
                    matches.append(
                        f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                    )
    assert matches == []
