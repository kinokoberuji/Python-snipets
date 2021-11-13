from __future__ import annotations
from typing import List, Dict

import seaborn as sns
import matplotlib.pyplot as plt

from collections.abc import Iterable, Iterator

# Simulate a pandas series of 100 observations of categorical data with 5 levels of values

def simulate_series(n: int, levels: int) -> pd.Series:
    """
    Simulate a pandas series of n observations of categorical data with levels levels of values
    """
    return pd.Series(np.random.randint(0, levels, n))

# Vận dụng mẫu thiết kế Iterator để tạo một phổ màu từ một tập hợp n keys

class ColorIterator(Iterator):

    """Class Iterator"""

    def __init__(self, key_lst: Color_Collection, palette: str = "husl") -> None:

        self.key_lst = key_lst
        self.index = 0
        self.colors = sns.color_palette(palette, len(key_lst)).as_hex()

    def __next__(self) -> Dict[str, str]:
        try:
            key = self.key_lst[self.index]
            color = self.colors[self.index]
            self.index += 1
            return {key: color}

        except IndexError:
            raise StopIteration()


class Color_Collection(Iterable):

    """Class quản lý tập hợp keys và tạo phổ màu"""

    def __init__(self, key_lst: List[str]) -> None:
        self.key_lst = key_lst

    def __iter__(self, *args, **kwargs) -> ColorIterator:
        return ColorIterator(self.key_lst, *args, **kwargs)

    def add_item(self, key: str) -> None:
        self.key_lst.append(key)

    def remove_var(self, key: str) -> None:
        self.key_lst.remove(key)

    def __len__(self) -> int:
        return len(self.key_lst)

    def get_color_palette(self, *args, **kwargs) -> Dict[str, str]:

        if not self.key_lst:
            raise ValueError("Lỗi: key list rỗng")

        color_dict = {}
        for color in ColorIterator(self.key_lst, *args, **kwargs):
            color_dict.update(color)

        return color_dict

    def show_color_palette(self, *args, **kwargs) -> None:
        color_dict = self.get_color_palette(*args, **kwargs)
        sns.palplot(sns.color_palette(list(color_dict.values())))
        plt.show()

    def __repr__(self) -> str:
        return f"Phổ màu cho tập hợp ({self.key_lst})"