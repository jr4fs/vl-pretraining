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
from beta_sampling import *
from random_sampling import *
from max_variability_sampling import * 
from min_variability_sampling import *
from max_confidence_sampling import * 
from min_confidence_sampling import * 
from global_random_sampling import *
from global_max_confidence_sampling import *
from global_min_confidence_sampling import *
from global_min_variability_sampling import * 
from global_max_variability_sampling import * 
from global_min_variability_sampling_multilabel import * 
from global_max_variability_sampling_multilabel import * 
from global_random_multilabel import *
from beta_sampling_multilabel import * 
from global_max_confidence_sampling_multilabel import *
from global_min_confidence_sampling_multilabel import *

if __name__ == "__main__":
    print("SEED sampler.py: ", args.seed)
    base_path = args.base_path
    print("base path: ", base_path)
    sampling_method = args.sampling_method
    print("sampling method: ", sampling_method)
    print("sampling dataset: ", args.sampling_dataset)
    df = pd.read_pickle(base_path+"datamap_metrics.pkl")
    #budgets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    budgets = [30]
    if sampling_method == 'beta':
        #beta_sampling(df, args.alpha, args.beta, args.sampling_model, args.training_budget, norm=args.norm, bandwidth=args.bandwidth, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
        params = [(1, 2), (2, 1), (1, 1),(2, 2)]
        #params = [(1, 1)]
        norms = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        # norms = ['pvals', 'var_counts', 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        #norms = ['exponential', 'gaussian', 'var_counts']
        for norm in norms:
            print("Norm: ", norm)
            for budget in budgets:
                for param in params:
                    df = pd.read_pickle(base_path+"datamap_metrics.pkl")
                    beta_sampling(df, param[0], param[1], args.sampling_model, budget, norm=norm, bandwidth=args.bandwidth, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'beta_multilabel':
        #params = [(1, 2), (2, 1), (1, 1),(2, 2)]
        params = [(1, 2), (2, 1), (1, 1),(2, 2), (8,2), (2,8)]
        #norms = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        norms = ['tophat', 'pvals']
        # norms = ['pvals', 'var_counts', 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        for norm in norms:
            print("Norm: ", norm)
            for budget in budgets:
                for param in params:
                    df = pd.read_pickle(base_path+"datamap_metrics.pkl")
                    beta_sampling_multilabel(df, param[0], param[1], args.sampling_model, budget, norm=norm, bandwidth=args.bandwidth, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'random':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            random_sampling(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'max_variability':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            max_variability(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'min_variability':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            min_variability(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'max_confidence':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            max_confidence(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'min_confidence':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            min_confidence(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_random':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_random_sampling(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_max_confidence':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_max_confidence(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_max_confidence_multilabel':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_max_confidence_multilabel(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_min_confidence':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_min_confidence(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_min_confidence_multilabel':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_min_confidence_multilabel(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_min_variability':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_min_variability(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_max_variability':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_max_variability(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_min_variability_multilabel':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_min_variability_multilabel(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_max_variability_multilabel':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_max_variability_multilabel(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    elif sampling_method == 'global_random_multilabel':
        for budget in budgets:
            df = pd.read_pickle(base_path+"datamap_metrics.pkl")
            global_random_multilabel(df, args.sampling_model, budget, include_all_classes=args.include_all_classes, dataset=args.sampling_dataset)
    else:
        print("Sampling method not implemented")