#!/usr/bin/env python
"""Build a standalone 40-frame human snippet library from SOP03 indexes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.long_snippet_library import (  # noqa: E402
    build_long_snippet_library,
    load_long_snippet_artifact,
    write_long_snippet_artifacts,
)
from src.datasets.motion_snippet_utils import (  # noqa: E402
    audit_snippet_source_overlap,
)
from src.datasets.split_manager import SPLIT_NAMES  # noqa: E402
from src.datasets.thor_adapter import (  # noqa: E402
    ThorDataError,
    load_recording_indexes_from_dir,
    load_recording_split_provenance,
)
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-crop standalone 40-frame human snippets from SOP03 indexes."
    )
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument(
        "--base-config", type=Path, default=_ROOT / "configs/base.yaml"
    )
    parser.add_argument("--stride-s", type=float, default=1.0)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    thresholds = base_config["dynamic_objects"]["human"]
    recordings_dir = args.recording_root / "recording_indexes" / args.split
    recordings = load_recording_indexes_from_dir(
        recordings_dir, expected_split=args.split
    )
    provenance = load_recording_split_provenance(recordings_dir)
    library = build_long_snippet_library(
        recordings,
        split=args.split,
        object_type="human",
        stride_s=args.stride_s,
        min_mean_speed_mps=float(thresholds["min_speed_mps"]),
        max_mean_speed_mps=float(thresholds["max_speed_mps"]),
        max_acceleration_mps2=float(thresholds["max_acceleration_mps2"]),
        split_provenance=provenance,
    )
    overlap_report = audit_snippet_source_overlap([library])
    if overlap_report["status"] != "ok":
        raise ThorDataError("long snippet source overlap detected")
    artifact_dir = args.output_dir / args.split / "human"
    paths = write_long_snippet_artifacts(
        library,
        artifact_dir,
        overlap_report=overlap_report,
    )
    load_long_snippet_artifact(paths["directory"])
    print(f"library[{args.split}/human]={paths['library']}")
    print(f"accepted_count[{args.split}/human]={library.summary['accepted_count']}")
    print(f"candidate_count[{args.split}/human]={library.summary['candidate_count']}")
    print(f"semantic_digest_sha256={library.summary['semantic_digest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
