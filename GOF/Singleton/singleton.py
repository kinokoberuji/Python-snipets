
from threading import Lock, Thread
from typing import List, Optional
import pandas as pd

class SingletonMeta(type):
    """
    Metaclass triển khai mô thức Singleton và tương thích với multithreading.
    """

    _instances = {}
    _lock: Lock = Lock()
    """
    Khi truy nhập lần đầu tiên, cho phép đồng bộ hóa các threads nhờ lock object
    """

    def __call__(cls, *args, **kwargs):
        # Khi chương trình thi hành lần đầu tiên, chưa có một object Experiment
        # nào cả, nhiều threads có thể được thi hành đồng thời, thread đầu tiên
        # sẽ được khóa (lock) và được thi hành cho đến cùng, trong khi những
        # thread còn lại sẽ chờ tại vị trí này.
        with cls._lock:
            # thread đầu tiên sẽ bị khóa và tiến vào trong đoạn code điều kiện bên dưới,
            # kiểm tra điều kiện và khởi tạo một object Experiment. Khi đã thoát
            # ra khỏi đoạn code bị khóa, những thread còn lại vốn đang chờ đợi
            # sẽ thi hành đồng thời các tiến trình sau đoạn code này;
            # nhưng vì class Experiment đã khởi tạo 1 lần,
            # nó sẽ không tạo thêm object nào mới nữa.

            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class Experiment(metaclass=SingletonMeta):
    _data: pd.DataFrame = None
    _exp = {}

    def __init__(self, path: str) -> None:
        if self._data is None:
            try:
                self._data = pd.read_csv(path, index_col=0)
            except FileNotFoundError:
                print(f"Lỗi: File {path} không tồn tại")
                self._data = None

    def create_experiment(self, exp_name: str,
                          X: Optional[List[str]] = None,
                          Y: Optional[List[str]] = None) -> None:

        """Hàm thiết lập 1 thí nghiệm mới"""

        if exp_name not in self._exp.keys():
            print(f"Tạo thí nghiệm mới tên là {exp_name}")
            print(f"Bao gồm tập biến X = {X}")
            print(f"và tập biến Y = {Y}")
            self._exp[exp_name] = {'X': X, 'Y': Y}
        else:
            print(f"Thí nghiệm {exp_name} đã tồn tại, hãy chọn tên khác cho thí nghiệm")

    def get_info(self, exp_name: str):

        """Hàm tóm tắt thông tin về dữ liệu cho 1 thí nghiệm"""

        if exp_name in self._exp.keys():
            X = self._exp[exp_name]['X']
            Y = self._exp[exp_name]['Y']
            df = self._data[X + Y]
            print(f"Thông tin của thí nghiệm {exp_name}")
            print(df.info())
            print(df.groupby(Y).describe().T)

        else:
            print(f"Lỗi: Thí nghiệm {exp_name} chưa tồn tại")

    def get_data(self, exp_name: str):

        """Hàm lấy dataframe cho 1 thí nghiệm"""

        if exp_name in self._exp.keys():
            X = self._exp[exp_name]['X']
            Y = self._exp[exp_name]['Y']
            df = self._data[X + Y]
            return df

        else:
            print(f"Lỗi: Thí nghiệm {exp_name} chưa được thiết lập")
            return None
    

