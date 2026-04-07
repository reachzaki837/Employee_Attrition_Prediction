import sys
from pathlib import Path

# ensure workspace root in path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
import pandas as pd

df = pd.read_csv(settings.PROCESSED_DATA_PATH)
print(df.columns.tolist())
print(df.head())
