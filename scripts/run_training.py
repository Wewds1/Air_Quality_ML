from src.config import RAW_DATA_PATH
from src.train import train_pipeline

summary = train_pipeline(RAW_DATA_PATH)
print(summary)