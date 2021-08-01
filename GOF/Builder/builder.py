from __future__ import annotations

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from abc import ABC, abstractmethod

class IGraph(ABC):

    @abstractmethod
    def basic_plot(self) -> None:
        pass

    @abstractmethod
    def add_2D_kde(self) -> None:
        pass

    @abstractmethod
    def add_ref_kde(self) -> None:
        pass

    @abstractmethod
    def add_bias_kde(self) -> None:
        pass

    @abstractmethod
    def draw_graph(self) -> None:
        pass

class BA_tool(IGraph):

    def __init__(self, df: pd.DataFrame, ref:str, targ:str):

        self.df = df
        self.df['Bias'] = self.df[ref] - self.df[targ]
        self.ref = ref
        self.targ = targ

        self.limits = {'xmin': min(self.df[ref])-0.5, 
                       'xmax':max(self.df[ref])+0.5,
                       'ymin':min(self.df['Bias'])-0.5, 
                       'ymax':max(self.df['Bias'])+0.5}
        
        self.ba_stats = self.df['Bias'].describe(percentiles = [0.05,0.5,0.95])[4:7].to_list()

        plt.rcParams["figure.figsize"] = (14,8)
        plt.rcParams.update({'font.size': 15})

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3])
        gs.update(hspace=0.09, wspace=0.09)

        self.fig = fig
        self.gs = gs

    def basic_plot(self) -> None:

        ba_df = self.df
        ba_stats = self.ba_stats
        ref = self.ref
            
        xmin, xmax, ymin, ymax = self.limits['xmin'], self.limits['xmax'],\
                 self.limits['ymin'], self.limits['ymax']

        fig, gs = self.fig, self.gs

        ax = plt.subplot(gs[1,0]) # Instantiate scatter plot area and axis range
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True)

        x = ba_df[ref].values
        y = ba_df['Bias'].values

        sns.scatterplot(x,
                        y,
                        ax = ax,
                        alpha = 0.9, 
                        color = 'black', 
                        s = 45,
                        label = None)

        ax.hlines(y = ba_stats + [0],
                xmin = xmin, 
                xmax = xmax,
                linestyles = 'dashed',
                color = 'black',
                alpha = 0.8)

        for i in ba_stats:
            ax.text(x = xmax, 
                        y = i,
                        s = f"{i:.3}",
                        color='k')

        ax.set_ylabel("Measurement bias")
        ax.set_xlabel("Reference scale")

        self.fig, self.gs, self.ax = fig, gs, ax
    
    def add_2D_kde(self) -> None:

        fig, gs, ax = self.fig, self.gs, self.ax
        ba_df = self.df
        
        ref = self.ref

        x = ba_df[ref].values
        y = ba_df['Bias'].values
    
        xmin, xmax, ymin, ymax = self.limits['xmin'], self.limits['xmax'],\
                 self.limits['ymin'], self.limits['ymax']

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.flatten(), yy.flatten()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        ax.contourf(xx, yy, f, 
                            level = 5, 
                            alpha = 0.4, 
                            extend = 'neither', 
                            colors = sns.color_palette("rainbow", 12)
                        )

        self.fig, self.gs, self.ax = fig, gs, ax

    def add_ref_kde(self) -> None:

        fig, gs,ax = self.fig, self.gs, self.ax
        ba_df = self.df
        
        ref = self.ref
        targ = self.targ

        x = ba_df[ref].values
        xp = ba_df[targ].values

        axl = plt.subplot(gs[0,0], sharex=ax) 
        axl.get_xaxis().set_visible(False) 
        axl.get_yaxis().set_visible(False)
        axl.spines["right"].set_visible(False)
        axl.spines["left"].set_visible(False)
        axl.spines["top"].set_visible(False)
        axl.spines["bottom"].set_visible(False)

        axl = sns.kdeplot(x, 
                      ax = axl, 
                      color = 'r', 
                      alpha = 0.6,
                      shade = True, 
                      vertical = False)

        axl = sns.kdeplot(xp, 
                        ax = axl, 
                        linestyle = '--',
                        color = 'k', 
                        alpha = 0.8,
                        shade = False, 
                        vertical = False)

        self.fig, self.gs = fig,  gs

    def add_bias_kde(self) -> None:

        fig, gs,ax = self.fig, self.gs, self.ax

        ba_df = self.df
        y = ba_df['Bias'].values
        ym = np.median(y)
        
        axb = plt.subplot(gs[1,1], sharey=ax) 
        axb.get_xaxis().set_visible(False) 
        axb.get_yaxis().set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.spines["left"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axb.spines["bottom"].set_visible(False)

        axb = sns.kdeplot(y, 
                      ax = axb, 
                      color = '#7303fc', 
                      shade = True, 
                      alpha = 0.5,
                      vertical = True)

        axb.hlines(y = ym, 
               xmin = 0, 
               xmax=0.02,
               linestyles = 'dashed',
               alpha = 0.8,
               color = 'k')

        self.fig = fig
    
    def draw_graph(self) -> None:

        plt.tight_layout(self.fig)
        plt.show(self.fig)

class Bland_Altman:
    
    def __init__(self, df: pd.DataFrame, ref:str, targ:str):
        self.ba_plot = BA_tool(df, ref, targ)

    def plot_basic_ba(self):
        self.ba_plot.basic_plot()
        self.ba_plot.draw_graph()

    def plot_bias_kde_ba(self):
        self.ba_plot.basic_plot()
        self.ba_plot.add_bias_kde()
        self.ba_plot.draw_graph()

    def plot_ref_kde_ba(self):
        self.ba_plot.basic_plot()
        self.ba_plot.add_ref_kde()
        self.ba_plot.draw_graph()

    def plot_full_1D_kde(self):
        self.ba_plot.basic_plot()
        self.ba_plot.add_ref_kde()
        self.ba_plot.add_bias_kde()
        self.ba_plot.draw_graph()

    def plot_2D_kde_ba(self):
        self.ba_plot.basic_plot()
        self.ba_plot.add_2D_kde()
        self.ba_plot.draw_graph()

    def plot_fancy_ba(self):
        self.ba_plot.basic_plot()
        self.ba_plot.add_2D_kde()
        self.ba_plot.add_ref_kde()
        self.ba_plot.add_bias_kde()
        self.ba_plot.draw_graph()