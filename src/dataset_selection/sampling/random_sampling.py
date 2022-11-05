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
from sklearn.neighbors import KernelDensity


def random_sampling(df, model, training_budget, dataset='animals'):
    targets = df['Target'].tolist()
    targets = [i[0] for i in targets]
    df['Target'] = targets
    orig_df = df.copy()
    all_ids = df['question_id'].to_list()
    num_total_samples = round(len(all_ids) * (training_budget * 0.01))

    sampled_question_ids = []
    sampled_variabilities = []
    sampled_confidence = []
    sampled_correctness = []
    sampled_targets = []

    unique_targets = df['Target'].unique()
    while len(sampled_question_ids) <= num_total_samples:
        for label in unique_targets:
            if len(sampled_question_ids) <= num_total_samples:
                # For each unique class, generate a variability distribution
                df_filtered = df[df['Target'] == label]
                question_ids = np.array(df_filtered['question_id'].to_list())
                variabilities = np.array(df_filtered['variability'].tolist())
                confidence = np.array(df_filtered['confidence'].tolist())
                correctness = np.array(df_filtered['correctness'].tolist())
                targets_sample = np.array(df_filtered['Target'].tolist())
                num_samples = round(len(question_ids) * (training_budget*0.01))

                if len(question_ids) !=0:
                    # for each bucket, sample the minimum number of examples from that bucket and append to overall sample
                    idx = np.random.choice(np.arange(len(question_ids)), num_samples, replace=False)
                    df = df.drop(df[df.question_id.isin(question_ids[idx])].index)
                    sampled_question_ids.extend(question_ids[idx])
                    sampled_variabilities.extend(variabilities[idx])
                    sampled_confidence.extend(confidence[idx])
                    sampled_correctness.extend(correctness[idx])
                    sampled_targets.extend(targets_sample[idx])

    ax = sns.displot(data=df, x='variability', kde=True).set(title=model+': Variability Distribution')
    ax2 = sns.displot(sampled_variabilities, kde=True).set(title=model+': Variability Distribution: \n Random Sampling, p='+str(training_budget))
    #ax2.set(ylim=(0, 250))

    fig, ax3 = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=sampled_variabilities, y=sampled_confidence, cmap="inferno", shade=False, thresh=0.05, n_levels=30, ax=ax3).set(title=model+": Random Sampling, Budget: " + str(training_budget))
    #sns.move_legend(ax3, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    unique_targets_random = set(sampled_targets)

    intersect = set(unique_targets) - unique_targets_random
    if len(intersect) != 0:
        for target_excluded in intersect:
            df_filtered = orig_df[orig_df['Target'] == target_excluded]
            sampled_question_ids.extend(df_filtered['question_id'].to_list())

    unique_targets_sample = orig_df[orig_df['question_id'].isin(sampled_question_ids)]
    unique_targets = set(unique_targets_sample['Target'].unique())
    save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/random/'+'train_budget_'+str(training_budget)+'.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(list(set(sampled_question_ids)), f)

    print("unique targets random: ", len(set(unique_targets)))
    print('samples - random: ', len(set(sampled_question_ids)))
    print('all_samples - random: ', len(all_ids))
    #return sampled_question_ids