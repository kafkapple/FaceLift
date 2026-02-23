"""Experiment and report configuration schema."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""

    name: str
    method: str  # "facelift" | "pose-splatter"
    pipeline: str  # "gslrm" | "e2e" | "per-scene"
    input_views: int
    metrics_json: str
    render_dir: str = ""
    render_pattern: str = "{fid}/render_view_{vid:02d}.png"
    color: str = "#333333"


@dataclass
class ProtocolConfig:
    """Evaluation protocol configuration."""

    name: str
    description: str
    source: str  # "overall" or "per_view"
    holdout_view: int = -1
    holdout_key: str = ""


@dataclass
class ReportConfig:
    """Full report configuration loaded from YAML."""

    title: str
    dataset: dict
    experiments: list[ExperimentConfig]
    protocols: dict[str, ProtocolConfig]
    metrics: dict
    visualization: dict

    @classmethod
    def from_yaml(cls, path: str) -> "ReportConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        experiments = [ExperimentConfig(**e) for e in data["experiments"]]

        protocols = {}
        for key, pdata in data.get("protocols", {}).items():
            protocols[key] = ProtocolConfig(**pdata)

        return cls(
            title=data["title"],
            dataset=data["dataset"],
            experiments=experiments,
            protocols=protocols,
            metrics=data.get("metrics", {"primary": [], "secondary": []}),
            visualization=data.get("visualization", {}),
        )
