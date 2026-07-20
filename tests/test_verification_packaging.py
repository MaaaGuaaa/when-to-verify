from pathlib import Path

import tomli


ROOT = Path(__file__).resolve().parents[1]


def test_verification_optional_dependency_and_package_discovery_are_declared():
    config = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["optional-dependencies"]["verification"] == [
        "torch>=2.0,<3"
    ]
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "src*"
    ]
