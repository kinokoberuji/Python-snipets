### ADAPTER sử dụng OOP và Inheritance
import os
import pandas as pd
import pyreadr


class CSV:
    """Quy trình mặc định tải file CSV"""

    def load_csv(self, f_name):
        return pd.read_csv(f_name)


class XLSX:
    """Quy trình tải file Excel"""

    def load_excel(self, f_name: str):
        return pd.read_excel(f_name)


class SPSS:
    """Quy trình tải file SPSS"""

    def load_spss(self, f_name):
        return pd.read_spss(f_name)


class RDATA:
    """Quy trình tải file RData"""

    def load_rdata(self, f_name):
        return pyreadr.read_r(f_name)["df"]


# Adapter class sử dụng multiple Inheritance
class Multi_Loader(CSV, SPSS, XLSX, RDATA):
    """Class tải dữ liệu, dùng 1 method load cho cả 3 định dạng"""

    def load(self, f_name):

        ext = os.path.splitext(f_name)[1]

        if (ext == ".csv") or (ext == ".txt"):
            return self.load_csv(f_name)
        elif ext == ".xlsx":
            return self.load_excel(f_name)
        elif ext == ".sav":
            return self.load_spss(f_name)
        elif ext == ".RData":
            return self.load_rdata(f_name)
        else:
            raise Exception(f"Không hỗ trợ định dạng {ext}")