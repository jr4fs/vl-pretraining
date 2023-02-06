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


def global_random_multilabel(df, model, training_budget, include_all_classes=False, dataset='animals'):
    #targets = df['Target'].tolist()
    #targets = [i[0] for i in targets]
    #df['Target'] = targets

    targets = df['Target'].tolist()
    targets = [i[0] for i in targets]
    df['Target'] = targets
    targets = df['Target'].tolist()

    targets_multilabel =[]
    for i in targets:
        target_list = [x.strip() for x in i.split(',')]
        targets_multilabel.extend(target_list)
    unique_targets = set(targets_multilabel)

    orig_df = df.copy()


    all_ids = np.array(df['question_id'].to_list())
    variabilities = np.array(df['variability'].tolist())
    confidence = np.array(df['confidence'].tolist())
    correctness = np.array(df['correctness'].tolist())
    targets_sample = np.array(df['Target'].tolist())

    num_total_samples = round(len(all_ids) * (training_budget * 0.01))

    sampled_question_ids = []
    sampled_variabilities = []
    sampled_confidence = []
    sampled_correctness = []
    sampled_targets = []

    
    idx = np.random.choice(np.arange(len(all_ids)), num_total_samples, replace=False)

    sampled_question_ids.extend(all_ids[idx])
    sampled_variabilities.extend(variabilities[idx])
    sampled_confidence.extend(confidence[idx])
    sampled_correctness.extend(correctness[idx])
    sampled_targets.extend(targets_sample[idx])

    sampled_targets_unique = []
    for sample in sampled_targets:
        target_list = [x.strip() for x in sample.split(',')]
        sampled_targets_unique.extend(target_list)
    sampled_targets_unique = set(sampled_targets_unique)

    targets_excluded = unique_targets - sampled_targets_unique
    print("TARGETS EXCLUDED: ", len(targets_excluded))
    
    samples_left = set(all_ids) - set(sampled_question_ids) # question ids not in sample

    idx_to_add = [] # indices of ids to add 
    unique_targets_sample = orig_df[orig_df['question_id'].isin(samples_left)] # question ids of examples not in original sample
    targets_excluded_sample = unique_targets_sample['Target'].tolist() # targets of examples not included in original sample 
    for excluded_target in targets_excluded: # for every excluded target, enumerate each example and check if the excluded target is in that example, if so, add it to original sample
        for idx_exclude, target_excluded_example in enumerate(targets_excluded_sample):
            if excluded_target in target_excluded_example:
                idx_to_add.append(idx_exclude)
                break

    ids_left = unique_targets_sample['question_id'].tolist()

    additional_targets = []
    for i in idx_to_add:
        additional_targets.append(ids_left[i])
    sampled_question_ids.extend(additional_targets)
    print("LEN: ", len(additional_targets))




    save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/global_random/seed_'+str(args.seed)+'/budget_'+str(training_budget)+'.pkl'


    # unique_targets_sample = orig_df[orig_df['question_id'].isin(sampled_question_ids)]
    # unique_targets = set(unique_targets_sample['Target'].unique())

    with open(save_path, 'wb') as f:
        pickle.dump(list(set(sampled_question_ids)), f)

    print("unique targets random: ", len(set(unique_targets)))
    print('samples - random: ', len(set(sampled_question_ids)))
    print('full train set: ', len(all_ids))
    #return sampled_question_ids