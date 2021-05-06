from __future__ import annotations
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score

import matplotlib.pyplot as plt
from xgboost import XGBClassifier # XGboost

from abc import ABC, abstractmethod

from typing import Dict, Set

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

xgb_mod = XGBClassifier(booster='dart',
                        tree_method = "gpu_hist",
                        n_estimators = 300,
                        learning_rate = 0.05,
                        predictor = 'gpu_predictor',
                        eval_metric = 'logloss',
                        max_depth = 3,
                        gpu_id = 0)

xgb_mod.load_model('CVD_mod')

cvd_df = pd.read_csv('cardio_train.csv', sep = ';', index_col=0)

cvd_df['age'] =  cvd_df['age'] / 365.24
cvd_df['gender'] = cvd_df['gender'] - 1

cvd_df = cvd_df[(cvd_df['ap_lo'] <= 370) & (cvd_df['ap_lo'] > 0)]
cvd_df = cvd_df[(cvd_df['ap_hi'] <= 370) & (cvd_df['ap_hi'] > 0)]
cvd_df = cvd_df[cvd_df['ap_hi'] >= cvd_df['ap_lo']]

cvd_df.reset_index(drop=True, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(cvd_df.drop(['cardio'], axis=1), cvd_df['cardio'], test_size=0.5, random_state=123)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

X_test.reset_index(inplace = True, drop = True)
y_test.reset_index(inplace = True, drop = True)

def add_noise(x: pd.DataSeries) -> pd.Series:
    return x + np.random.normal(0.5,1,x.shape[0])

class Service(ABC):
    """
    The Service interface declares a set of methods for managing subscribers.
    """

    @abstractmethod
    def attach(self, client: Client) -> None:
        """+
        Attach an client to the service.
        """
        pass

    @abstractmethod
    def detach(self, client: Client) -> None:
        """
        Detach an client from the service.
        """
        pass

    @abstractmethod
    def notify(self) -> None:
        """
        Notify all clients about an event.
        """
        pass

    @abstractmethod
    def visualize_states(self) -> None:
        """Visualize model's performances
        """
        pass


class ConcreteService(Service):
    """
    The Service owns some important state and notifies clients when the state
    changes.
    """
    
    _newdata = {'X': pd.DataFrame(), 'y': pd.DataFrame()}
    _state = {'f1': [], 'bac': []}
    """
    For the sake of simplicity, the Service's state, essential to all
    subscribers, is stored in this variable.
    """

    _clients: Set(Client) = set()
    """
    List of subscribers. In real life, the list of subscribers can be stored
    more comprehensively (categorized by event type, etc.).
    """

    def attach(self, client: Client) -> None:
        print(style.GREEN + f"Khách hàng mới {client._name} thuê bao dịch vụ")
        self._clients.add(client)

    def detach(self, client: Client) -> None:
        print(style.GREEN + f"Khách hàng {client._name} chấm dứt sử dụng dịch vụ")
        self._clients.remove(client)

    """
    The subscription management methods.
    """

    def notify(self) -> None:
        """
        Trigger an update in each subscriber.
        """
        print(style.BLACK + "Cập nhật kết quả kiểm định cho tất cả khách hàng ...")
        print(style.GREEN + pd.DataFrame(self._state).iloc[[-1]].to_string(index=False))

        for client in self._clients:
            client.update(self)
        
        self._newdata = {'X': pd.DataFrame(), 'y': pd.DataFrame()}

    def model_following_up(self, xgb_mod: XGBClassifier) -> None:
        """
        Usually, the subscription logic is only a fraction of what a Service can
        really do. Services commonly hold some important business logic, that
        triggers a notification method whenever something important is about to
        happen (or after it).
        """

        temp_X = self._newdata['X']
        temp_y = self._newdata['y']

        pred = xgb_mod.predict(temp_X)
        bac = balanced_accuracy_score(temp_y, pred)
        f1 = f1_score(temp_y, pred)
        self._state['f1'].append(f1)
        self._state['bac'].append(bac)

        print(style.BLACK + f"Mô hình đang được kiểm tra định kì trên {len(temp_X)} bệnh nhân mới ...")

        self.notify()
    
    def visualize_states(self) -> None:
        plt.plot(self._state['f1'], '-r', alpha = 0.7, label = 'F1')
        plt.plot(self._state['bac'], '-b', alpha = 0.5, label = 'BAC')
        plt.hlines(y = 0.7, xmin = 1, xmax = len(self._state['f1']),linestyles='dashed')
        plt.legend()
        plt.xlabel('Lượt kiểm định')
        plt.ylabel('Hiệu năng')
        plt.show()

class Client(ABC):
    """
    The Client interface declares the update method, used by services.
    """
    @abstractmethod
    def data_transfer(self, service: Service) -> None:
        """Send new data to Service
        """
        pass

    @abstractmethod
    def update(self, service: Service) -> None:
        """
        Receive update from service.
        """
        pass


"""
Concrete Clients react to the updates issued by the Service they had been
attached to.
"""

class ConcreteClientA(Client):

    _name = 'A'

    def data_transfer(self, service: Service) -> None:
        print(style.BLUE + f"Khách hàng A cập nhật dữ liệu mới cho dịch vụ ...")
        spl_idx = np.random.choice(X_test.index, np.random.choice(range(50,81)))
        temp_y = y_test.loc[spl_idx]
        temp_X = X_test.loc[spl_idx]

        temp_X.loc[:,['age','height','weight','ap_hi','ap_lo']] = \
            temp_X[['age','height','weight','ap_hi','ap_lo']].apply(lambda x: add_noise(x))

        service._newdata['X'] = pd.concat([service._newdata['X'], temp_X])
        service._newdata['y'] = pd.concat([service._newdata['y'], temp_y])

    def update(self, service: Service) -> None:
        if service._state['f1'][-1] < 0.7:
            print(style.BLUE + f"Khách hàng A: báo động xuất hiện F1 = {service._state['f1'][-1]}")
        else:
            print(style.BLUE + 'Khách hàng A: F1 của mô hình ổn định')

class ConcreteClientB(Client):

    _name = 'B'

    def data_transfer(self, service: Service) -> None:
        print(style.RED + f"Khách hàng B cập nhật dữ liệu mới cho dịch vụ ...")
        spl_idx = np.random.choice(X_test.index,np.random.choice(range(50,81)))
        temp_y = y_test.loc[spl_idx]
        temp_X = X_test.loc[spl_idx]

        temp_X.loc[:,['age','height','weight','ap_hi','ap_lo']] = \
            temp_X[['age','height','weight','ap_hi','ap_lo']].apply(lambda x: add_noise(x))

        service._newdata['X'] = pd.concat([service._newdata['X'], temp_X])
        service._newdata['y'] = pd.concat([service._newdata['y'], temp_y])

    def update(self, service: Service) -> None:
        if service._state['bac'][-1] < 0.7:
            print(style.RED +f"Khách hàng B: báo động xuất hiện BAC = {service._state['bac'][-1]}")
        else:
            print(style.RED + 'Khách hàng B: BAC của mô hình ổn định')
