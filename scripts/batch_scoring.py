from src.config import RAW_DATA_PATH
from src.score_batch import score_batch

scored = score_batch(RAW_DATA_PATH)
print(scored.head())