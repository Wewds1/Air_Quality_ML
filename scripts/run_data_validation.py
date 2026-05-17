from src.config import RAW_DATA_PATH
from src.data_loading import load_raw_data
from src.validation import validation_summary

df = load_raw_data(RAW_DATA_PATH)
print(validation_summary(df))