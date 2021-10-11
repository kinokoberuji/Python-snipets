import os
import pandas as pd
import pyreadr


def load_rdata(f_name: str) -> pd.DataFrame:
    """Hàm tải file rdata"""
    return pyreadr.read_r(f_name)["df"]


# Sử dụng Python dictionary
LOADERS = {
    ".csv": pd.read_csv,
    ".txt": pd.read_csv,
    ".sav": pd.read_spss,
    ".xlsx": pd.read_excel,
    ".RData": load_rdata,
}

# class và hàm thực ra không khác nhau xa
class Loader:
    def __call__(self, f_name: str) -> pd.DataFrame:
        ext = os.path.splitext(f_name)[1]
        return LOADERS[ext](f_name)


# Sử dụng 1 hàm duy nhất
def multi_load(f_name: str) -> pd.DataFrame:
    """Hàm cho phép tải cả 3 định dạng data"""
    ext = os.path.splitext(f_name)[1]
    try:
        return LOADERS[ext](f_name)
    except:
        print(f"Lỗi: không thể tải file {f_name}")
        return None


# Sử dụng decorator
def adapt_extension(func):
    def load(**kwargs):
        f_name = kwargs["f_name"]
        ext = os.path.splitext(f_name)[1]
        return LOADERS[ext](f_name)

    return load


@adapt_extension
def load_data(f_name):
    print(f"Đã tải file {f_name}")