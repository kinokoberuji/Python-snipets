
import pandas as pd

def dtype_based_rule(data: pd.DataFrame) -> bool:

    return (data.dtype == "object") or (data.dtype == "string")

def unique_val_based_rule(data: pd.DataFrame) -> bool:

    return len(data.unique()) <= 5