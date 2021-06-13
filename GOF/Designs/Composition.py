from __future__ import annotations

# Tools
import pandas as pd
import numpy as np
import pickle
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

# OOP
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

SIG_COLS = ['#650bdb','#db0b50','#e62e09']
ENV_COLS = ['#6cb518', '#ff8503']

LABELS = {2: 'Gắng sức hô hấp',
          3: 'Giảm thở tắc nghẽn',
          4: 'Giảm thở trung ương',
          5: 'Ngưng thở tắc nghẽn',
          6: 'Ngưng thở hỗn hợp',
          7: 'Ngưng thở trung ương'}

@dataclass
class Signal(ABC): 
    """Mẫu class dữ liệu
    """
    sig_path: str

    @abstractmethod
    def load_signal(self):
        """Tải dữ liệu từ file pickle"""

@dataclass
class Events(ABC): 

    """Mẫu class biến cố
    """
    evt_path: str

    @abstractmethod
    def load_evt(self):
        """Tải dữ liệu từ file pickle"""

@dataclass
class Raw_signal(Signal):
    """Class tín hiệu
    """
    def load_signal(self):
        with open(self.sig_path, 'rb') as handle:
            sig_df = pickle.load(handle)
        return sig_df['Raw_signal']

@dataclass
class Evt_lst(Events):
    """Class danh sách biến cố
    """
    def load_evt(self):
        with open(self.evt_path, 'rb') as handle:
            evt_lst = pickle.load(handle)

        return evt_lst['PSG_score']

@dataclass
class Envelop:
    """Class dữ liệu biên độ dao động
    """
    _sig_df: Raw_signal

    @staticmethod
    def envelop(x: pd.Series, w: int, mode: str):
        """Xác định ngưỡng biên độ tín hiệu
        """
        ts = pd.Series(x.values)

        if mode == 'up':
            peaks, _ = find_peaks(ts, width = w)
        elif mode == 'down':
            peaks, _ = find_peaks(-ts, width = w)
        
        ts_b = ts.copy()
        ts_b.iloc[~ts_b.index.isin(peaks)] = np.nan

        ts_b = ts_b.interpolate(method='linear', limit_direction='forward')
        ts_b = ts_b.fillna(method  = 'bfill')
                                    
        return ts_b
    
    def multi_envelop(self, w: int):
        """Xác định ngưỡng biên độ tín hiệu
        """
        df = self._sig_df.load_signal()

        ub = df.apply(self.envelop, axis = 0, mode = 'up', w = w)
        lb = df.apply(self.envelop, axis = 0, mode = 'down', w = w)
        ub.index = df.index
        lb.index = df.index

        return lb, ub

@dataclass
class Sample:
    """Class trích xuất đoạn tín hiệu tại idx
    """
    idx: int
    _evt_lst : Evt_lst

    def get_sample(self):

        evt_lst = self._evt_lst.load_evt()
        start_t = evt_lst.iloc[self.idx]['start_t']
        end_t = evt_lst.iloc[self.idx]['end_t']

        return start_t, end_t

@dataclass
class Signal_viz:

    """Class vẽ biểu đồ
    """
    _sig_df: Raw_signal
    _evt_lst: Evt_lst
    _envelop: Optional[Envelop]
    _sample: Sample
    idx: int
    
    def sample(self,df: pd.DataFrame):

        """Trích xuất phân đoạn thứ i"""
        start_t, end_t = self._sample.get_sample()
        temp_raw = df.loc[start_t:end_t,:]

        return temp_raw

    def visualize(self, mode = 'Acc'):

        """Vẽ biểu đồ
        """
        raw_df = self._sig_df.load_signal()
        idx = self.idx
        evt_lst = self._evt_lst.load_evt()
        lb, ub = self._envelop.multi_envelop(w = 10)
        
        target = evt_lst.iloc[idx]['Lab']
        label = LABELS[target]
        channels = list(raw_df.columns)

        temp_raw = self.sample(raw_df)
        temp_down = self.sample(lb)
        temp_up = self.sample(ub)

        fig, axs = plt.subplots(nrows=3,
                                ncols=1,
                                sharex=True, 
                                sharey=False,
                                dpi = 100,
                                figsize=(10,6))
        
        axs[0].set_title(f"Phân đoạn {idx}: {label}")
        
        axs[2].set_xlabel('Thời gian')

        for row,k in enumerate(channels):

            axs[row].set_ylabel(k)
            axs[row].plot(temp_raw[k],
                        alpha = 0.7,
                        lw=1.2,
                        color = SIG_COLS[row])

            if mode == 'Acc':
                axs[row].plot(temp_up[k], 
                      alpha = 1,
                      lw=1,
                      color = ENV_COLS[1])

                axs[row].plot(temp_down[k], 
                            alpha = 1,
                            lw=1,
                            color = ENV_COLS[0])
        
        plt.tight_layout()
        plt.show()


