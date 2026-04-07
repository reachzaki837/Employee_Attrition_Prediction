import sys
from pathlib import Path

# ensure workspace root is on path so `src` package can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data, get_data_summary
from src.data.cleaner import clean_pipeline
from src.data.validator import validate_clean_data


def main():
    df = load_raw_data()
    print("raw shape", df.shape)
    print("summary", get_data_summary(df))
    df_clean = clean_pipeline(df)
    print("clean shape", df_clean.shape)
    validate_clean_data(df_clean)
    print("validation passed")


if __name__ == "__main__":
    main()
