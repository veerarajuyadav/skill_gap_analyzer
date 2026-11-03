"""Central path and configuration management for the AcadLink Skill Gap project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Top-level directory layout
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"
TESTS_DIR = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

ONTOLOGY_DIR = RAW_DIR / "ontology"
ONET_DIR = ONTOLOGY_DIR / "onet"
ESCO_DIR = ONTOLOGY_DIR / "esco"

MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "_cache"


def ensure_dirs() -> None:
    for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FINAL_DIR, MODELS_DIR,
              EXPERIMENTS_DIR, REPORTS_DIR, ONTOLOGY_DIR, ONET_DIR, ESCO_DIR,
              MODEL_CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42

# Ontology sources
ESCO_VERSION = "v1.2.1"
ONET_VERSION = "30.3"
