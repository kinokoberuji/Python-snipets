from typing import Callable
from categorical_checkers import unique_val_based_rule
from dataclasses import dataclass, field
from enum import Enum, auto
import pandas as pd

class DType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()

Check_Dtype = Callable[[pd.DataFrame, str], bool]
    
@dataclass
class Data:

    data: pd.DataFrame = field(init=True)
    dtype: DType = DType.NUMERIC
    _group: str = field(default = None)
    method: Check_Dtype = unique_val_based_rule
    
    def set_group(self, group: str)->None:
        self._group = group
        
    def set_dtype(self, dtype: DType)->None:
        self.dtype = dtype
        
    def set_method(self, method: Check_Dtype)->None:
        self.method = method
        
    @property
    def filtered(self) -> pd.DataFrame:

        df = self.data.copy()
        check_method = self.method

        if self.dtype == DType.NUMERIC:
            return df.loc[:,~df.apply(lambda x: check_method(x))]
        elif self.dtype == DType.CATEGORICAL:
            return df.loc[:,df.apply(lambda x: check_method(x))]
        else:
            return df[[]]
        
    @property
    def group(self) -> str:
        return self._group
    
    @property
    def stratified(self) -> pd.DataFrame:
        
        if self._group:
            return self.data[[self._group]]
        else:
            return self.data[[]]