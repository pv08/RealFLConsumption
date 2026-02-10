from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
PECANSTREET_DIR = DATASET_DIR / "pecanstreet"
FREQ_15MIN_DIR = PECANSTREET_DIR / "15min"
AUSTIN_DIR = FREQ_15MIN_DIR / "austin"

TRAIN_DIR = AUSTIN_DIR / "train"
TEST_DIR = AUSTIN_DIR / "test"