import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re 
import seaborn as sns
import math 
import collections
from pycocotools.coco import COCO
import requests
import plotly.express as px
# from jupyter_dash import JupyterDash
# from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from os import listdir
from os.path import isfile, join
import base64
import itertools
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')

from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import operator
import plotly.figure_factory as ff
import scipy
import pickle 
from param import args

def scatter_it_sampled(dataframe, hue_metric ='correct.', title='', model='LXMERT', show_hist=False, sampled_question_ids_path=None):
    # Subsample data to plot, so the plot is not too busy.
    #dataframe = dataframe.sample(n=25000 if dataframe.shape[0] > 25000 else len(dataframe))
    print(sampled_question_ids_path)

    with open(sampled_question_ids_path, 'rb') as f:
        sampled_ids = pickle.load(f)
    all_ids = dataframe['question_id'].tolist()
    remaining_ids = set(all_ids) - set(sampled_ids)
    
    df_one = dataframe.loc[dataframe['question_id'].isin(sampled_ids)]
    df_two = dataframe.loc[dataframe['question_id'].isin(remaining_ids)]
    concatenated = pd.concat([df_two.assign(dataset='remaining'), df_one.assign(dataset='sampled')])


    # Normalize correctness to a value between 0 and 1.
    #dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    #dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]
    dataframe['correct.'] = dataframe['correctness']
    
    main_metric = 'variability'
    other_metric = 'confidence'
    
    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        ax0 = axs
    else:
        fig = plt.figure(figsize=(16, 10), )
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])
    
        ax0 = fig.add_subplot(gs[0, :])
    
    
    ### Make the scatterplot.
    
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")
    pal.reverse()

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=concatenated,
                           palette=['lightblue', 'darkblue'],
                           hue='dataset',
                           style='dataset',
                           s=30)
    
    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    an1 = ax0.annotate("ambiguous", xy=(0.9, 0.5), xycoords="axes fraction", fontsize=15, color='black',
                  va="center", ha="center", rotation=350, bbox=bb('black'))
    an2 = ax0.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction", fontsize=15, color='black',
                  va="center", ha="center", bbox=bb('r'))
    an3 = ax0.annotate("hard-to-learn", xy=(0.35, 0.25), xycoords="axes fraction", fontsize=15, color='black',
                  va="center", ha="center", bbox=bb('b'))
    
    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=(1.01, 0), loc='center left', fancybox=True, shadow=True)
    else:
        plot.legend(fancybox=True, shadow=False,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')
    plot.set_title(f"{model}-{title} Data Map", fontsize=17)

    fig.tight_layout()
    fig_save = fig.get_figure()
    sample_name = os.path.basename(os.path.splitext(sampled_question_ids_path)[0])
    fig_save.savefig(args.base_path+sample_name+'_datamap.pdf') 


if __name__ == "__main__":
    df = pd.read_pickle(args.base_path+"datamap_metrics.pkl")
    print(args.base_path)
    #sampled_question_ids_path = '../src/dataset_selection/sampling/samples/LXR111/myo-sports/beta/beta_kernel/linear/seed_388/alpha_2_beta_2_budget_30.pkl'
    scatter_it_sampled(df, title=' Beta-Linear a=2, b=2, Budget: 30,', show_hist=True, sampled_question_ids_path=args.sampling_ids)
