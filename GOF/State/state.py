from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod

from typing import List
from dataclasses import dataclass, field

import os

# System call
os.system("")

# Class of different styles
class style():
    BLACK = '\u001b[30m'
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    BLUE = '\u001b[34m'
    RESET = '\033[0m'

def response(y0:float, t: float, ke: float):
    return y0*np.exp(ke*t)*(1-0.012)-0.018

class IState(ABC):
    "Mẫu State"

    @staticmethod
    @abstractmethod
    def __call__():
        "Set the default method"

class Lab_test(IState):
    "Xét nghiệm định lượng Marker X"

    @staticmethod
    def test(patient: Clinical_context):
        "Xét nghiệm"

        t_i = len(patient.marker)
        if t_i == 0:
            x = float(np.random.gamma(1.5,0.5,1))
            patient.marker = x * 30
            print(style.RED + f"Xét nghiệm lần đầu tiên, kết quả = {x*30: .3f} mUI/mL")

        else:
            y0 = patient.marker[0]
            yt = patient.marker[-1]
            dif = 100*(yt - y0)/y0

            if dif <= -99.99:
                print(style.RESET + "Kết quả xét nghiệm đã âm tính, kết thúc theo dõi")
                patient.treated = True
            else:
                ke = patient.clearance
                y0 = patient.marker[0]
                x = response(y0 = y0, t = t_i, ke = ke)
                patient.marker = x
                print(style.BLUE + f"Xét nghiệm lần {t_i + 1}, kết quả = {x: .3f} mUI/mL")
        
        print("Trả kết quả xét nghiệm cho bác sĩ điều trị")

    __call__ = test

class Treatment(IState):
    "Quyết định lâm sàng"

    @staticmethod
    def make_decision(patient: Clinical_context):
        "Can thiệp tùy theo kết quả CLS"

        t_i = len(patient.marker)
        y0 = patient.marker[0]
        yt = patient.marker[-1]
        ke = patient.clearance
        t = len(patient.marker)
        dose = patient.dose
        dif = 100*(yt - y0)/y0

        if t_i == 1:
            patient.dose += 0.01
            print(f"Bắt đầu hóa trị, liều {dose:.2f} mg")
        
        if dif< -95.0:
            print(style.RESET + "Chấm dứt điều trị, bệnh nhân đã khỏi bệnh")
            patient.treated = True
        
        else:
            print(style.GREEN + f"Đáp ứng điều trị tuần thứ {t_i} = {dif:.3f} %, tiếp tục theo dõi")

            if (dif > -50) & (t_i >= 3):
                print(style.RED + "Bệnh nhân không đáp ứng liều thấp, tăng liều gấp đôi")
                patient.dose *= 2
        
        ke -= patient.dose * 0.85
        patient.clearance = ke

    __call__ = make_decision

@dataclass
class Clinical_context():
    """Tình huống lâm sàng của một bệnh nhân"""

    _marker: List[float] = field(default_factory=list)
    _clearance: float = -0.1
    _dose: float = 0.01
    treated: bool = False

    _state_handles = [Lab_test(),
                      Treatment()]

    _handle = iter(_state_handles)

    @property
    def marker(self):
        return self._marker

    @marker.setter
    def marker(self, level):
        self._marker.append(level)
    
    @property
    def clearance(self):
        return self._clearance
    
    @clearance.setter
    def clearance(self, Ke):
        self._clearance = Ke
    
    @property
    def dose(self):
        return self._dose
    
    @dose.setter
    def dose(self, new_dose):
        self._dose = new_dose

    def request(self):
        try:
            self._handle.__next__()(patient = self)
        except StopIteration:
            # resetting so it loops
            self._handle = iter(self._state_handles)
    
    def followup(self):

        if len(self.marker) > 2:
            ts = np.array(range(1,len(self.marker)+1))
            sns.lineplot(x = np.array(ts), 
                    y = np.array(self.marker), 
                    color = 'red')
            plt.ylabel('Biomarker X (mUI/mL)')
            plt.xlabel('Thời gian theo dõi (tuần)')
            plt.show()
        else:
            print("Chưa đủ dữ liệu !")