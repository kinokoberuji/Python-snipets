### ADAPTER sử dụng OOP và Composition

from __future__ import annotations
import os
import pandas as pd
from typing import Optional
import pyreadr


class CSV:
    """Quy trình mặc định tải file CSV"""

    def load(self, f_name):
        return pd.read_csv(f_name)


class XLSX:
    """Quy trình tải file Excel"""

    def load(self, f_name):
        return pd.read_excel(f_name)


class SPSS:
    """Quy trình tải file SPSS"""

    def load(self, f_name):
        return pd.read_spss(f_name)


class RDATA:
    """Quy trình tải file RData"""

    def load(self, f_name):
        return pyreadr.read_r(f_name)["df"]


# Adapter class sử dụng Composition
F_TYPES = {
    ".txt": CSV(),
    ".csv": CSV(),
    ".xlsx": XLSX(),
    ".sav": SPSS(),
    ".RData": RDATA(),
}


class Special_loader(object):
    """Adapter class sử dụng Composition"""

    def __init__(
        self,
        f_type: Optional[
            CSV,
        ] = None,
    ):
        self.f_type = f_type

    def load(self, f_name) -> pd.DataFrame:
        try:
            return self.f_type.load(f_name)
        except:
            print(f"Lỗi: không thể tải được {f_name}")

class Adapt_Loader:
    """Class tải dữ liệu, dùng 1 method load cho cả 3 định dạng"""

    def __init__(self,f_type: Optional[CSV] = None):
        self.f_type = f_type
        
    def spec_load(self, f_name: str):
        try:
            return self.f_type.load(f_name)
        except:
            print(f"Lỗi: không thể tải được {f_name}")
    
    def load(self, f_name: str):

        ext = os.path.splitext(f_name)[1]
        self.f_type = F_TYPES[ext]
        
        return self.spec_load(f_name)