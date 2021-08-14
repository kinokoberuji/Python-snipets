from __future__ import annotations

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

def koopman_tlco(x : np.array):

    '''Mô hình Koopman (2011), dành cho trẻ em từ 5-18 tuổi
    ước lượng 3 tham số Mu, Lambda, Sigma của DLCO theo phân phối Gamma chuẩn hóa
    '''

    sex = x[0]
    A = x[1]
    H = x[2]
    
    if sex == 1:
        M = (np.exp(34.8048-6.8925*np.log(H)-8.6557*np.log(12*A)+0.1043+1.7893*np.log(H)*np.log(12*A)))/0.3348
        L = np.exp(-1.63-0.0348*A)
        S = 2/(1/(np.exp(-1.63-0.0348*A))**2)**0.5
    else:
        M = (np.exp(34.8048-6.8925*np.log(H)-8.6557*np.log(12*A)+1.7893*np.log(H)*np.log(12*A)))/0.3348
        L = np.exp(-2.38+0.0276*A)
        S = 2/(1/(np.exp(-1.63-0.0348*A))**2)**0.5
    
    return M,L,S

def bts_ers_tlco(x : np.array):

    '''Mô hình BTS-ERS (1993) của Cotes và Quanjer,
    ước lượng trung bình dự báo Mu của LDCO theo phân phối chuẩn
    Áp dụng cho người lớn từ 18-75 tuổi
    '''
    
    sex = x[0]
    A = x[1]
    H = x[2]
        
    if sex == 1:
        M = 0.325*H-0.2*A-17.6
        se = 5.1
    else:
        M = 0.212*H-0.156*A-2.66
        se = 3.69

    ULN = M + 1.96 * se
    rsd = (ULN - M)/1.96
        
    return M,rsd

def RSD_Z_score(feats):

    '''Hàm tính điểm Z-score dựa vào Mu và RSD
    '''
    M = feats[0]
    rsd = feats[1]
    obs = feats[2]
    
    Z= (obs-M)/rsd
    
    return Z

def LMS_Z_score(feats):

    '''Hàm tính điểm Z-score dựa vào 3 tham số Mu, Lambda, Sigma
    '''

    M = feats[0]
    L = feats[1]
    S = feats[2]
    obs = feats[3]
                
    Z=((obs/M)**L-1)*(1/(L*S))

    return Z

@dataclass
class TLCO_interpreter:

    _strategy: Strategy = field(init=True)
    data: pd.DataFrame = field(init=True)

    @property
    def strategy(self) -> Strategy:
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def transform(self, targ:str) -> pd.DataFrame:

        data = self.data
        output = self.strategy.interpret(df = data, targ=targ)
        return output

class Strategy(ABC):

    @abstractmethod
    def interpret(self, df: pd.DataFrame, targ:str) -> pd.Series:
        pass

class Child_strategy(Strategy):
    
    def interpret(self, df: pd.DataFrame, targ: str) -> pd.Series:
        
        input_df = df[['Sex','Age','Height']]
        pred_df = input_df.apply(koopman_tlco, axis=1)

        pred_df = pd.DataFrame([[*t] for t in pred_df.values], columns = ['M','L','S'])
        pred_df['Obs'] = df[targ].values

        Z_df = pred_df.apply(LMS_Z_score, axis = 1)

        return Z_df

class Adult_strategy(Strategy):

    def interpret(self, df: pd.DataFrame, targ: str) -> pd.Series:
        
        input_df = df[['Sex','Age','Height']]

        pred_df = input_df.apply(bts_ers_tlco,axis = 1)
        pred_df = pd.DataFrame([[*t] for t in pred_df.values], columns = ['M','RSD'])

        pred_df['Obs'] = df[targ]

        Z_df = pred_df.apply(RSD_Z_score, axis = 1)

        return Z_df