from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data_dir: Path = ROOT / "DryBeanDataset"
    artifacts: Path = ROOT / "artifacts"
    data_artifacts: Path = ROOT / "artifacts" / "data"
    results: Path = ROOT / "artifacts" / "results"
    figures: Path = ROOT / "artifacts" / "figures"
    models: Path = ROOT / "artifacts" / "models"
    screenshots: Path = ROOT / "artifacts" / "screenshots"
    paper: Path = ROOT / "paper"


PATHS = Paths()
RANDOM_SEED = 42
TARGET = "Class"
RAW_FEATURES = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
)


def ensure_artifact_dirs() -> None:
    for path in (
        PATHS.artifacts,
        PATHS.data_artifacts,
        PATHS.results,
        PATHS.figures,
        PATHS.models,
        PATHS.screenshots,
        PATHS.paper,
    ):
        path.mkdir(parents=True, exist_ok=True)

