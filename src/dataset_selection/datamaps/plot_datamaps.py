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


def scatter_it(dataframe, hue_metric ='correct.', title='', model='LXMERT', show_hist=False):
    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=25000 if dataframe.shape[0] > 25000 else len(dataframe))
    
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
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
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


    # fig = px.scatter(dataframe, x=main_metric,
    #                        y=other_metric, 
    #                        color=hue,
    #                        symbol=style,
    #                        size_max=30)
    # fig.show()

    
    if show_hist:
        plot.set_title(f"{model}-{title} Data Map", fontsize=17)
        
        # Make the histograms.
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal', range=[0,1])
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')

        #plot2 = sns.countplot(x="correct.", data=dataframe, color='#86bf91', ax=ax3)
        plot2 = dataframe.hist(column=['correct.'], ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True) # Show the vertical gridlines

        plot2[0].set_title('')
        plot2[0].set_xlabel('correctness')
        plot2[0].set_ylabel('')

    fig.tight_layout()
    fig_save = fig.get_figure()
    fig_save.savefig(args.base_path+'datamap.pdf') 

def generate_class_variability_distributions(base_path, df, dataset='animals'):
    # if region == 'hard':
    #     # segment instances thresholded on confidence and variability to separate regions on datamap 
    #     outliers = df.loc[(df['confidence'] < conf_threshold) & (df['variability'] < var_threshold)]
    # elif region == 'easy':
    #     outliers = df.loc[(df['confidence'] > conf_threshold) & (df['variability'] < var_threshold)]
    # else:
    #     outliers = df.loc[(df['confidence'] > conf_threshold) & (df['variability'] > var_threshold) & (df['confidence'] < conf_threshold_two)]
    targets = df['Target'].tolist()
    targets = [i[0] for i in targets]
    df['Target'] = targets
    unique_targets = df['Target'].unique()
    
    n_cols = 6  
    n_rows = math.ceil(len(unique_targets) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=(unique_targets))
    counter = 0
    for row in range(n_rows):
        for col in range(n_cols):
            target_df = df[df['Target'] == unique_targets[counter]]
            fig.add_trace(go.Histogram(histfunc="count", x=target_df['variability'], showlegend=False), row=row+1, col=col+1)
            counter+=1
            if counter == len(unique_targets):
                break
    fig.update_xaxes(range=[0,0.5])
    fig.update_layout(height=2000)
    fig.show()
    fig.write_image(file=base_path+dataset+'_target_variability_distribution.pdf', format='pdf')

def plot_trainval_acc(base_path):
    '''
    Plots train/val accuracy scores 

            Parameters:
                    base_path (str): Path to model metadata

    '''
    with open(args.base_path + 'log.log') as fp:
        acc = fp.readlines()
    train_scores = []
    valid_scores = []
    for i in acc:
        if 'Train' in i:
            train_scores.append(float(i[-7:].strip()))
        elif 'Valid' in i:
            valid_scores.append(float(i[-7:].strip()))

    xs_valid = [i for i in range(len(valid_scores))]
    xs_train = [i for i in range(len(train_scores))]
    plt.plot(xs_valid, valid_scores, label="Validation")
    #plt.title("Validation")
    #plt.savefig(base_path+'/training.png')

    plt.plot(xs_train, train_scores, label="Training")
    #plt.title("Training")
    plt.savefig(args.base_path+'/train_val.png')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Correct Preds")

if __name__ == "__main__":
    df = pd.read_pickle(args.base_path+"datamap_metrics.pkl")
    plot_trainval_acc(args.base_path)
    scatter_it(df, title=args.datamap_title, show_hist=True)