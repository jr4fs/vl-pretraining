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
from param import args

def global_max_confidence_multilabel(df, model, training_budget, include_all_classes=False, dataset='animals'):
    targets = df['Target'].tolist()
    targets = [i[0] for i in targets]
    df['Target'] = targets
    orig_df = df.copy()

    sampled_question_ids = []
    sampled_variabilities = []
    sampled_confidence = []
    sampled_correctness = []
    sampled_targets = []

    unique_targets = df['Target'].unique()
    

    df_sorted = df.sort_values(by=['confidence'], ascending=False)
    question_ids = np.array(df_sorted['question_id'].to_list())
    variabilities = np.array(df_sorted['variability'].tolist())
    confidence = np.array(df_sorted['confidence'].tolist())
    correctness = np.array(df_sorted['correctness'].tolist())
    targets_sample = np.array(df_sorted['Target'].tolist())

    num_total_samples = round(len(question_ids) * (training_budget * 0.01))

    sampled_question_ids.extend(question_ids[:num_total_samples])
    sampled_variabilities.extend(variabilities[:num_total_samples])
    sampled_confidence.extend(confidence[:num_total_samples])
    sampled_correctness.extend(correctness[:num_total_samples])
    sampled_targets.extend(targets_sample[:num_total_samples])

    # ax = sns.displot(data=df, x='variability', kde=True).set(title=model+': Variability Distribution')
    # ax2 = sns.displot(sampled_variabilities, kde=True).set(title=model+': Variability Distribution: \n Random Sampling, p='+str(training_budget))
    # #ax2.set(ylim=(0, 250))

    # fig, ax3 = plt.subplots(figsize=(10, 5))
    # sns.kdeplot(x=sampled_variabilities, y=sampled_confidence, cmap="inferno", shade=False, thresh=0.05, n_levels=30, ax=ax3).set(title=model+": Random Sampling, Budget: " + str(training_budget))
    # #sns.move_legend(ax3, "upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # if include_all_classes == True:
    #     unique_targets_random = set(sampled_targets)
    #     intersect = set(unique_targets) - unique_targets_random
    #     if len(intersect) != 0:
    #         for target_excluded in intersect:
    #             df_filtered = orig_df[orig_df['Target'] == target_excluded]
    #             sampled_question_ids.extend(df_filtered['question_id'].to_list())
    #     save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/global_max_confidence/include_all_classes/seed_'+str(args.seed)+'/budget_'+str(training_budget)+'.pkl'
    # else:
    #     save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/global_max_confidence/seed_'+str(args.seed)+'/budget_'+str(training_budget)+'.pkl'
    save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/global_max_confidence/seed_'+str(args.seed)+'/budget_'+str(training_budget)+'.pkl'


    # unique_targets_sample = orig_df[orig_df['question_id'].isin(sampled_question_ids)]
    # unique_targets = set(unique_targets_sample['Target'].unique())


    targets_multilabel =[] # unique targets in vqa dataset 
    for i in targets_sample:
        target_list = [x.strip() for x in i.split(',')]
        targets_multilabel.extend(target_list)
    unique_targets = set(targets_multilabel)

    sampled_targets_unique = [] # unique targets in sampled data
    for sample in sampled_targets:
        target_list = [x.strip() for x in sample.split(',')]
        sampled_targets_unique.extend(target_list)
    sampled_targets_unique = set(sampled_targets_unique)
    targets_excluded = unique_targets - sampled_targets_unique
    print("TARGETS EXCLUDED: ", len(targets_excluded))


    with open(save_path, 'wb') as f:
        pickle.dump(list(set(sampled_question_ids)), f)

    print("unique targets global max confidence ", len(set(sampled_targets_unique)))
    print('samples - global max confidence: ', len(set(sampled_question_ids)))
    print('all_samples - global max confidence: ', len(question_ids))
    #return sampled_question_ids