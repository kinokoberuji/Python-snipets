from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

SIG_COLS = ['#650bdb','#db0b50','#e62e09']
ENV_COLS = ['#6cb518', '#ff8503']

LABELS = {2: 'Gắng sức hô hấp',
          3: 'Giảm thở tắc nghẽn',
          4: 'Giảm thở trung ương',
          5: 'Ngưng thở tắc nghẽn',
          6: 'Ngưng thở hỗn hợp',
          7: 'Ngưng thở trung ương'}

def load_data(file_path: str):
    """Tải dữ liệu từ file pickle
    """
    try:
        with open(file_path, 'rb') as handle:
            data = pickle.load(handle)
    except:
        data = None
    return data

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

def multi_envelop(df: pd.DataFrame, w: int):
    """Xác định ngưỡng biên độ tín hiệu
    """
    ub = df.apply(envelop, axis = 0, mode = 'up', w = w)
    lb = df.apply(envelop, axis = 0, mode = 'down', w = w)
    ub.index = df.index
    lb.index = df.index
    return lb, ub

def sample(raw_df: pd.DataFrame, 
           evt_lst: pd.DataFrame,
           idx: int):

    """Trích xuất phân đoạn thứ i"""

    start_t = evt_lst.iloc[idx]['start_t']
    end_t = evt_lst.iloc[idx]['end_t']
    temp_raw = raw_df.loc[start_t:end_t,:]

    return temp_raw

def visualize_signal(raw_df: pd.DataFrame, 
                    evt_lst: pd.DataFrame, 
                    idx: int,
                    visible = True):

    """Hiển thị dữ liệu polygraphy
    """
    target = evt_lst.iloc[idx]['Lab']
    label = LABELS[target]
    channels = list(raw_df.columns)

    temp_raw = sample(raw_df, evt_lst,idx)

    fig, axs = plt.subplots(nrows=3,
                            ncols=1,
                            sharex=True, 
                            sharey=False,
                            dpi = 120,
                            figsize=(10,6))
    
    axs[0].set_title(f"Phân đoạn {idx}: {label}")
    
    axs[2].set_xlabel('Thời gian')

    for row,k in enumerate(channels):

        axs[row].set_ylabel(k)
        axs[row].plot(temp_raw[k],
                      alpha = 0.7,
                      lw=1.2,
                      color = SIG_COLS[row])
    
    if visible:
        plt.tight_layout()
        plt.show()

    else:
        return fig, axs

def visualize_envelop(ub:pd.DataFrame,
                     lb: pd.DataFrame,
                     raw_df: pd.DataFrame,
                     evt_lst: pd.DataFrame, 
                     idx :int,
                     ):
    
    """Hiển thị dữ liệu Accelerometer"""

    channels = list(raw_df.columns)
    temp_up = sample(ub, evt_lst,idx)
    temp_down = sample(lb, evt_lst,idx)

    fig, axs = visualize_signal(raw_df, 
                                evt_lst, 
                                idx,
                                visible = False)
    
    for row,k in enumerate(channels):
        
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