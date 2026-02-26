from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
