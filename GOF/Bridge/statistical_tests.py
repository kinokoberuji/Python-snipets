import pandas as pd
import pingouin as pg

def chi_sq_test(data: pd.DataFrame, x: str,y: str)-> str:

    """Chi-square test"""

    _, _, stats = pg.chi2_independence(data,x,y)
    
    return f"{stats.loc[1,'pval']}"

def t_test(data: pd.DataFrame,var: str,g: str) -> str:

    """Student independent samples t-test"""
    
    gp_data = data.groupby(g)[var].apply(lambda v: v.values)
    x = gp_data.iloc[0]
    y = gp_data.iloc[1]
    
    p = pg.ttest(x, y).iloc[0,3]
    
    return f"{p}"

def mwu_test(data: pd.DataFrame,var: str,g: str) -> str:

    """Mann-Whitney U test"""
    
    gp_data = data.groupby(g)[var].apply(lambda v: v.values)
    x = gp_data.iloc[0]
    y = gp_data.iloc[1]
    
    p = pg.mwu(x, y).iloc[0,2]
    
    return f"{p}"