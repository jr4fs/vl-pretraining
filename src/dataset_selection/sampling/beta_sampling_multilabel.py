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
import multiprocessing

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))



def beta_sampling_multilabel(df, alpha, beta, model, training_budget, norm='pvals', bandwidth= 0.01, include_all_classes=False, dataset='animals'):
    # kernel = {'gaussian', 'tophat', epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’} 
    print("SEED beta.py: ", args.seed)
    targets = df['Target'].tolist()
    targets = [i[0] for i in targets]
    df['Target'] = targets
    
    all_ids = df['question_id'].to_list()
    num_total_samples = round(len(all_ids) * (training_budget * 0.01))
    question_ids = np.array(df['question_id'].tolist())
    variabilities = np.array(df['variability'].tolist()) * 2
    confidence = np.array(df['confidence'].tolist())
    targets = np.array(df['Target'].tolist())

    beta_distribution = scipy.stats.beta(alpha, beta)
    p_vals = beta_distribution.pdf(variabilities)
    #plt.plot(variabilities, p_vals, label='pdf')
    if norm == 'pvals':
        p_vals /= p_vals.sum()
        plt.plot(variabilities, p_vals, label='pdf')
        if include_all_classes==True:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/include_all_classes/beta_pvals/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
        else:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/beta_pvals/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
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
        if include_all_classes == True:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/include_all_classes/beta_var_counts/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
        else:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/beta_var_counts/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
    elif norm == 'gaussian_kde':
        kernel = scipy.stats.gaussian_kde(variabilities)
        print("bandwidth: ", kernel.factor)
        gaussian_eval = kernel.pdf(variabilities)
        fig1, ax1 = plt.subplots()
        p_vals /= gaussian_eval
        p_vals /= p_vals.sum()
        if include_all_classes == True:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/include_all_classes/beta_kernel/'+norm+'/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
        else:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/beta_kernel/'+norm+'/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
    elif norm in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
        print("1")
        vars_kde =np.array(variabilities).reshape(-1, 1)
        print("2")
        kde = KernelDensity(bandwidth=bandwidth, kernel=norm, atol=0.0005, rtol=0.01)
        print("3")
        kde.fit(vars_kde)
        print("4")
        #kde = KernelDensity(bandwidth=2.0,atol=0.0005,rtol=0.01).fit(sample) 
        logprob = parrallel_score_samples(kde, vars_kde)
        print("5")

        # score_samples returns the log of the probability density
        #logprob = kde.score_samples(vars_kde)
        p_vals /= np.exp(logprob)
        p_vals /= p_vals.sum()
        if include_all_classes == True:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/include_all_classes/beta_kernel/'+norm+'/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
        else:
            save_path = 'src/dataset_selection/sampling/samples/'+model+'/'+dataset+'/beta/beta_kernel/'+norm+'/seed_'+str(args.seed)+'/alpha_'+str(alpha)+'_beta_'+str(beta)+'_budget_'+str(training_budget)+'.pkl'
    else:
        print('Norm not implemented')

    
    idx = np.random.choice(np.arange(len(question_ids)), num_total_samples, replace=False, p=p_vals)
    print("8")
    sampled_question_ids = list(question_ids[idx])
    print("9")
    sampled_targets = list(targets[idx])
    print("10")
    # ax = sns.displot(data=df, x='variability', kde=True).set(title=model+': Variability Distribution')
    # ax2 = sns.displot(variabilities[idx], kde=True).set(title=model+': Variability Distribution \n Beta Distribution Sampling, Alpha=' + str(alpha) + ', Beta=' + str(beta))
    # fig, ax3 = plt.subplots(figsize=(10, 5))
    # sns.kdeplot(x=variabilities[idx], y=confidence[idx], cmap="inferno", shade=False, thresh=0.05, n_levels=30, ax=ax3).set(title=model+': Beta Distribution Variability Sampling, Alpha=' + str(alpha) + ', Beta=' + str(beta))
    # plt.show()

    # plt.figure(figsize=(50,10))
    # chart = sns.countplot(x=targets[idx], palette='Set1')
    # chart.set_title('Target Distribution: ' + model+': Beta Distribution Variability Sampling, Alpha=' + str(alpha) + ', Beta=' + str(beta))
    # chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    # figure = chart.get_figure()
    # figure.tight_layout()
    targets_multilabel =[] # unique targets in vqa dataset 
    for i in targets:
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

    # unique_targets_beta = set(targets[idx])
    # unique_targets = set(df['Target'].unique())
    # sampled_question_ids = list(question_ids[idx])
    # if include_all_classes == True:
    #     intersect = unique_targets - unique_targets_beta
    #     if len(intersect) != 0:
    #         for target_excluded in intersect:
    #             df_filtered = df[df['Target'] == target_excluded]
    #             sampled_question_ids.extend(df_filtered['question_id'].to_list())

    # unique_targets_sample = df[df['question_id'].isin(sampled_question_ids)]
    # unique_targets = set(unique_targets_sample['Target'].unique())
    #print("unique targets beta: ", len(set(targets[idx])))


    print("unique targets beta: ", len(unique_targets))
    print('samples - beta: ', len(set(sampled_question_ids)))
    print('all samples- beta sampling: ', len(all_ids))
    
    with open(save_path, 'wb') as f:
        pickle.dump(list(set(sampled_question_ids)), f)

    #return sampled_question_ids