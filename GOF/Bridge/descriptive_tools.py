import numpy as np
from scipy.stats import sem

def median_iqr(x):

    """Descriptive method for continous data,
    using Median and interquartile range
    """
    mu = x.median()
    l, u = np.quantile(x, [0.25, 0.75])

    return f"{mu:.2f}({l:.2f} - {u:.2f})"


def median_p95(x):

    """Descriptive method for continous data,
    using Median and 5th, 95th percentiles
    """
    mu = x.median()
    l, u = np.quantile(x, [0.05, 0.95])

    return f"{mu:.2f}({l:.2f} - {u:.2f})"


def mean_sd(x):

    """Descriptive method for continous data,
    using Mean and standard deviation
    """
    mu = x.mean()
    sigma = x.std()

    return f"{mu:.2f} ± {sigma:.2f}"


def mean_se(x):

    """Descriptive method for continous data,
    using Mean and standard error
    """
    mu = x.mean()
    se = sem(x)

    return f"{mu:.2f},({se:.2f})"


def freq_perct(x):

    """Descriptive method for categorical data,
    using frequency and percent
    """

    freq = x.sum()
    rate = x.mean()

    return f"{freq}({rate*100:.2f} %)"