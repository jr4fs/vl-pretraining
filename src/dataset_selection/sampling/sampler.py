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

if __name__ == "__main__":
    base_path = args.base_path
    sampling_method = args.sampling_method
    df = pd.read_pickle(base_path+"datamap_metrics.pkl")
    if sampling_method == 'beta':
        beta_sampling(df, args.alpha, args.beta, args.sampling_model, args.training_budget, norm=args.norm, bandwidth=args.bandwidth, dataset=args.sampling_dataset)
        # budgets = [10, 20, 30]
        # params = [(1, 1), (2, 2), (1, 2), (2, 1)]
        # norms = ['pvals', 'var_counts', 'gaussian_kde', 'cosine', 'epanechnikov', 'exponential', 'gaussian', 'linear', 'tophat']
        # for norm in norms:
        #     print("Norm: ", norm)
        #     for budget in budgets:
        #         for param in params:
        #             df = pd.read_pickle(base_path+"datamap_metrics.pkl")
        #             beta_sampling(df, param[0], param[1], args.sampling_model, budget, norm=norm, bandwidth=args.bandwidth, dataset=args.sampling_dataset)
    elif sampling_method == 'random':
        random_sampling(df, args.sampling_model, args.training_budget, dataset=args.sampling_dataset)
    else:
        print("Sampling method not implemented")