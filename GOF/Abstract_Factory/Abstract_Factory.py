import pandas as pd
from typing import Protocol, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter, WeibullAFTFitter


@dataclass
class Explorer(Protocol):

    data: pd.DataFrame = field(init=True)

    def data_viz(self, time_col: str, event_col: str) -> None:
        """Vẽ biểu đồ khảo sát dữ liệu Thời gian/Sự kiện"""


@dataclass
class Kaplan_Meier_Curve:

    data: pd.DataFrame = field(init=True)

    def data_viz(self, time_col: str, event_col: str) -> None:
        """Vẽ biểu đồ Kaplan-Meier"""
        data = self.data.copy()

        kmf = KaplanMeierFitter().fit(
            data[time_col], data[event_col], label="Kaplan-Meier"
        )
        kmf.plot_survival_function(color="#e60e56")
        plt.title("Biểu đồ Kaplan-Meier")
        plt.xlabel("Thời gian (ngày")
        plt.ylabel("Xác suất sống còn tích lũy")
        plt.show()


@dataclass
class Weibull_Curve:

    data: pd.DataFrame = field(init=True)

    def data_viz(self, time_col: str, event_col: str) -> None:
        """Vẽ biểu đồ Kaplan-Meier"""
        data = self.data.copy()

        wf = WeibullFitter().fit(data[time_col], data[event_col], label="Weibull")
        wf.plot_survival_function(color="#680fa8")
        plt.title("Đồ thị hàm survival với phân phối Weibull")
        plt.xlabel("Thời gian (ngày")
        plt.ylabel("Xác suất sống còn tích lũy")
        plt.show()


@dataclass
class Survival_model(Protocol):

    data: pd.DataFrame = field(init=True)

    def fit_model(self, time_col: str, event_col: str) -> None:
        """Khớp mô hình survival"""

    def summary(self) -> None:
        """Tóm tắt mô hình"""

    def plot_survival_function(self, idx: int) -> None:
        """Vẽ biểu đồ khảo sát dữ liệu Thời gian/Sự kiện"""


@dataclass
class CoxPH:

    data: pd.DataFrame = field(init=True)
    model = CoxPHFitter()

    def fit_model(self, time_col: str, event_col: str) -> None:
        """Khớp mô hình hồi quy CoxPH"""

        data = self.data.copy()
        self.model.fit(data, time_col, event_col)

    def summary(self) -> None:
        try:
            self.model.print_summary()
            self.model.plot()
            plt.show()
        except:
            print("Cần khớp Mô hình CoxPH với dữ liệu trước")

    def plot_survival_function(self, idx: int) -> None:
        try:
            X = self.data.iloc[idx]
        except IndexError:
            print(f"Trường hợp {idx} không tồn tại trong dữ liệu")

        try:
            self.model.predict_survival_function(X).rename(columns={idx: "CoxPH"}).plot(
                color="#e60e56"
            )
            plt.title(f"Biểu đồ hồi quy CoxPH với dữ liệu {idx}")
            plt.xlabel("Thời gian (ngày")
            plt.ylabel("Xác suất sống còn tích lũy")
            plt.show()
        except:
            print("Cần khớp Mô hình CoxPH với dữ liệu trước")


@dataclass
class Weibull_model:

    data: pd.DataFrame = field(init=True)
    model = WeibullAFTFitter()

    def fit_model(self, time_col: str, event_col: str) -> None:
        """Khớp mô hình hồi quy với phân phối Weibull"""

        data = self.data.copy()
        self.model.fit(data, time_col, event_col)

    def summary(self) -> None:
        try:
            self.model.print_summary()
            self.model.plot()
            plt.show()
        except:
            print("Cần khớp Mô hình Weibull với dữ liệu trước")

    def plot_survival_function(self, idx: int) -> None:

        try:
            X = self.data.iloc[idx]
        except IndexError:
            print(f"Trường hợp {idx} không tồn tại trong dữ liệu")

        try:
            self.model.predict_survival_function(X).rename(
                columns={idx: "Weibull"}
            ).plot(color="#680fa8")
            plt.title(f"Biểu đồ hồi quy Weibull với dữ liệu {idx}")
            plt.xlabel("Thời gian (ngày")
            plt.ylabel("Xác suất sống còn tích lũy")
            plt.show()
        except:
            print("Cần khớp Mô hình Weibull với dữ liệu trước")


CONFIGS = {
    "coxph": (Kaplan_Meier_Curve, CoxPH),
    "weibull": (Weibull_Curve, Weibull_model),
}


def generate_methods(data: pd.DataFrame) -> Tuple[Explorer, Survival_model]:
    """Khởi tạo 2 object khảo sát dữ liệu và mô hình survival"""

    while True:
        configs = input(f"Hãy chọn 1 trong 2 phương pháp {', '.join(CONFIGS)}: ")

        try:
            (curve, model) = CONFIGS[configs]
            return (curve(data), model(data))
        except KeyError:
            print(f"Tên phương pháp {configs} không hợp lệ !")


def survival_analysis(
    methods: Tuple[Explorer, Survival_model], time_col: str, event_col: str
):
    """Thực hiện phân tích Survival với method thùy chọn"""

    curve, model = methods
    curve.data_viz(time_col, event_col)
    model.fit_model(time_col, event_col)

    return model