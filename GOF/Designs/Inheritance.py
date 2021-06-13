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

SIG_COLS = ['#650bdb','#db0b50','#e62e09']
ENV_COLS = ['#6cb518', '#ff8503']

LABELS = {2: 'Gắng sức hô hấp',
          3: 'Giảm thở tắc nghẽn',
          4: 'Giảm thở trung ương',
          5: 'Ngưng thở tắc nghẽn',
          6: 'Ngưng thở hỗn hợp',
          7: 'Ngưng thở trung ương'}

@dataclass
class Data_pack(ABC): 

    """Mẫu class dữ liệu
    """
    sig_path: str
    evt_path: str

    @abstractmethod
    def load_signal(self):
        """Tải dữ liệu từ file pickle"""

    @abstractmethod
    def load_evt(self):
        """Tải dữ liệu từ file pickle"""

@dataclass
class PG_data(Data_pack):

    def load_signal(self):
        with open(self.sig_path, 'rb') as handle:
            sig_df = pickle.load(handle)
        self._sig_df = sig_df['Raw_signal']

    def load_evt(self):
        with open(self.evt_path, 'rb') as handle:
            evt_lst = pickle.load(handle)
        self._evt_lst = evt_lst['PSG_score']

@dataclass
class Acc_data(Data_pack):

    def load_signal(self):
        with open(self.sig_path, 'rb') as handle:
            sig_df = pickle.load(handle)
        self._sig_df = sig_df['Raw_signal']

    def load_evt(self):
        with open(self.evt_path, 'rb') as handle:
            evt_lst = pickle.load(handle)
        self._evt_lst = evt_lst['PSG_score']
    
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
        df = self._sig_df.copy()

        ub = df.apply(self.envelop, axis = 0, mode = 'up', w = w)
        lb = df.apply(self.envelop, axis = 0, mode = 'down', w = w)
        ub.index = df.index
        lb.index = df.index

        self._lb, self._ub = lb, ub
    
@dataclass
class PG_Visual(PG_data):

    idx: int

    def sample(self, df: pd.DataFrame):
        """Trích xuất phân đoạn thứ i"""
        start_t = self._evt_lst.iloc[self.idx]['start_t']
        end_t = self._evt_lst.iloc[self.idx]['end_t']
        temp_raw = df.loc[start_t:end_t,:]

        return temp_raw

    def visualize_signal(self):
        """Hiển thị dữ liệu PG
        """
        self.load_signal()
        self.load_evt()

        raw_df = self._sig_df
        idx = self.idx
        evt_lst = self._evt_lst
        
        target = self._evt_lst.iloc[idx]['Lab']
        label = LABELS[target]
        channels = list(raw_df.columns)

        temp_raw = self.sample(raw_df)

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
        
        plt.tight_layout()
        plt.show()

@dataclass
class Acc_Visual(Acc_data):

    idx: int

    def sample(self, df: pd.DataFrame):
        """Trích xuất phân đoạn thứ i"""
        start_t = self._evt_lst.iloc[self.idx]['start_t']
        end_t = self._evt_lst.iloc[self.idx]['end_t']
        temp_raw = df.loc[start_t:end_t,:]

        return temp_raw

    def visualize_signal(self):
        """Hiển thị dữ liệu Acc
        """
        self.load_signal()
        self.load_evt()

        idx = self.idx
        raw_df = self._sig_df

        self.multi_envelop(w = 10)            
        
        target = self._evt_lst.iloc[idx]['Lab']
        label = LABELS[target]
        channels = list(raw_df.columns)

        temp_raw = self.sample(df = raw_df)
        temp_down = self.sample(df = self._lb)
        temp_up = self.sample(df = self._ub)

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
    
         


