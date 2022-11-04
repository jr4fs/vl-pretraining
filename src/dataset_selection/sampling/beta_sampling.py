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


def beta_sampling(df, alpha, beta, model, training_budget, norm='pvals', bandwidth= 0.01, dataset='animals'):
    # kernel = {'gaussian', 'tophat', epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’} 
    targets = df['Target'].tolist()
    targets = [i[0] for i in targets]
    df['Target'] = targets
    
    all_ids = df['question_id'].to_list()
    num_total_samples = round(len(all_ids) * (training_budget * 0.01))
    question_ids = np.array(df['question_id'].tolist())
    variabilities = np.array(df['variability'].tolist())
    confidence = np.array(df['confidence'].tolist())
    targets = np.array(df['Target'].tolist())

    beta_distribution = scipy.stats.beta(alpha, beta)
    p_vals = beta_distribution.pdf(variabilities)
    #plt.plot(variabilities, p_vals, label='pdf')
    if norm == 'pvals':
        p_vals /= p_vals.sum()
        plt.plot(variabilities, p_vals, label='pdf')
        save_path = 'src/dataset_selection/sampling/samples/beta/beta_pvals/'+model+'_'+ dataset+'_alpha_'+str(alpha) + '_beta_'+str(beta) +'_' + str(training_budget)+'.pkl'
    elif norm == 'var_counts':
        # normalize by counts in variability histogram
        test = px.histogram(df, x='variability')
        f = test.full_figure_for_development(warn=False)
        xbins = f.data[0].xbins
        plotbins = list(np.arange(start=xbins['start'], stop=xbins['end']+xbins['size'], step=xbins['size']))
        counts, bins = np.histogram(list(f.data[0].x), bins=plotbins)
        #bins_quad = len(bins)//3
        #upper = bins[-bins_quad:]
        var_counts = []
        var_bins = np.digitize(variabilities, bins) # if it belongs to upper 25% of bins, weight it higher
        for var_bin in var_bins:
            #if var_bin in upper:
                #var_counts.append(counts[var_bin-1]*0.0000000005)
            #else:
            var_counts.append(counts[var_bin-1])
        var_counts /= sum(var_counts)
        p_vals /= var_counts
        p_vals /= p_vals.sum()
        save_path = 'src/dataset_selection/sampling/samples/beta/beta_var_counts/'+model+'_'+ dataset+'_alpha_'+str(alpha) + '_beta_'+str(beta) +'_' + str(training_budget)+'.pkl'
    elif norm == 'gaussian_kde':
        kernel = scipy.stats.gaussian_kde(variabilities)
        print("bandwidth: ", kernel.factor)
        gaussian_eval = kernel.pdf(variabilities)
        fig1, ax1 = plt.subplots()
        p_vals /= gaussian_eval
        p_vals /= p_vals.sum()
        save_path = 'src/dataset_selection/sampling/samples/beta/beta_kernel/'+model+'_'+ dataset+'_alpha_'+str(alpha) + '_beta_'+str(beta) +'_'+norm+'_' + str(training_budget)+'.pkl'
    elif norm in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
        vars_kde =np.array(variabilities).reshape(-1, 1)
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(vars_kde)
        # score_samples returns the log of the probability density
        logprob = kde.score_samples(vars_kde)
        p_vals /= np.exp(logprob)
        p_vals /= p_vals.sum()
        save_path = 'src/dataset_selection/sampling/samples/beta/beta_kernel/'+model+'_'+ dataset+'_alpha_'+str(alpha) + '_beta_'+str(beta) +'_'+norm+'_' + str(training_budget)+'.pkl'
    else:
        print('Norm not implemented')

    idx = np.random.choice(np.arange(len(question_ids)), num_total_samples, replace=False, p=p_vals)
    
    ax = sns.displot(data=df, x='variability', kde=True).set(title=model+': Variability Distribution')
    ax2 = sns.displot(variabilities[idx], kde=True).set(title=model+': Variability Distribution \n Beta Distribution Sampling, Alpha=' + str(alpha) + ', Beta=' + str(beta))
    fig, ax3 = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=variabilities[idx], y=confidence[idx], cmap="inferno", shade=False, thresh=0.05, n_levels=30, ax=ax3).set(title=model+': Beta Distribution Variability Sampling, Alpha=' + str(alpha) + ', Beta=' + str(beta))
    plt.show()

    plt.figure(figsize=(50,10))
    chart = sns.countplot(x=targets[idx], palette='Set1')
    chart.set_title('Target Distribution: ' + model+': Beta Distribution Variability Sampling, Alpha=' + str(alpha) + ', Beta=' + str(beta))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    figure = chart.get_figure()
    figure.tight_layout()

    unique_targets_beta = set(targets[idx])
    unique_targets = set(df['Target'].unique())

    intersect = unique_targets - unique_targets_beta
    sampled_question_ids = list(question_ids[idx])
    if len(intersect) != 0:
        for target_excluded in intersect:
            df_filtered = df[df['Target'] == target_excluded]
            sampled_question_ids.extend(df_filtered['question_id'].to_list())

    unique_targets_sample = df[df['question_id'].isin(sampled_question_ids)]
    unique_targets = set(unique_targets_sample['Target'].unique())
    #print("unique targets beta: ", len(set(targets[idx])))
    print("unique targets beta: ", len(unique_targets))
    print('samples - beta: ', len(set(sampled_question_ids)))
    print('all samples- beta sampling: ', len(all_ids))
    
    with open(save_path, 'wb') as f:
        pickle.dump(list(set(sampled_question_ids)), f)

    #return sampled_question_ids