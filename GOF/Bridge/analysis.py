from data_builder import Check_Dtype, DType

from dataclasses import dataclass, field
from typing import Callable, Protocol, List

import pandas as pd

Desc_Method = Callable[[pd.DataFrame], str]
Test_Method = Callable[[pd.DataFrame, str, str], str]


class Input_data(Protocol):
    def set_dtype(self, dtype: DType) -> None:
        ...

    def set_group(self, group: str) -> None:
        ...

    def set_method(self, method: Check_Dtype) -> None:
        ...

    @property
    def group(self) -> str:
        ...

    @property
    def stratified(self) -> pd.DataFrame:
        ...


class Explorer(Protocol):
    @property
    def mix_df(self) -> pd.DataFrame:
        ...

    def describe(self, desc_method: Desc_Method) -> pd.DataFrame:
        ...

    def test(self, test_method: Test_Method) -> pd.DataFrame:
        ...


@dataclass
class Categorical_explorer:

    data: Input_data = field(init=True)

    @property
    def mix_df(self) -> pd.DataFrame:

        df = self.data.filtered.copy()

        mix_df = pd.get_dummies(df.loc[:, self.variables], prefix_sep="=")

        mix_df = pd.concat([mix_df, self.data.stratified], axis=1)

        return mix_df

    @property
    def variables(self) -> List[str]:

        vars_lst = set(self.data.filtered.columns)
        vars_lst = vars_lst.difference(set([self.data.group]))

        return list(vars_lst)

    def test(self, test_method: Test_Method) -> pd.DataFrame:

        test_df = self.data.filtered
        grp_name = self.data.group
        pvals = dict()

        for x in self.variables:

            p = test_method(test_df, x, grp_name)

            pvals[x] = p

        return pd.DataFrame.from_dict(pvals, orient="index", columns=["p_val"])

    def describe(self, desc_method: Desc_Method) -> pd.DataFrame:

        mix_df = self.mix_df

        targs = mix_df.iloc[:, :-1].columns

        grp_name = self.data.group

        desc_table = mix_df.groupby(grp_name)[targs].agg(desc_method).T

        return desc_table


@dataclass
class Numeric_explorer:

    data: Input_data = field(init=True)

    @property
    def mix_df(self) -> pd.DataFrame:

        mix_df = pd.concat([self.data.filtered, self.data.stratified], axis=1)

        return mix_df

    def test(self, test_method: Test_Method) -> pd.DataFrame:

        mix_df = self.mix_df
        grp_name = self.data.group
        vars_lst = self.data.filtered.columns

        pvals = dict()

        for var in vars_lst:
            p = test_method(mix_df, var, grp_name)

            pvals[var] = p

        return pd.DataFrame.from_dict(pvals, orient="index", columns=["p_val"])

    def describe(self, desc_method: Desc_Method) -> pd.DataFrame:

        mix_df = self.mix_df
        grp_name = self.data.group
        vars_lst = self.data.filtered.columns

        desc_table = mix_df.groupby(grp_name)[vars_lst].agg(desc_method).T

        return desc_table