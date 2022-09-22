
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import math 
import collections
import fiftyone as fo
import fiftyone.zoo as foz
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from os import listdir
from os.path import isfile, join
import base64
import itertools

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')



def load_datamap_stats(base_path):
    '''
    Returns datamap stats recorded during training split by epochs

            Parameters:
                    base_path (str): Path to model metadata

            Returns:
                    df (DataFrame): Pandas dataframe with datamap stats: 
                                    confidence, variability and correctness
    '''
    # load logged stats from training 
    with open(base_path + 'datamaps_stats.json') as fp:
        datamap_stats_raw = json.load(fp)

    result = collections.defaultdict(list)
    for stat in datamap_stats_raw:
        result[stat['Epoch']].append(stat)
    datamap_stats = list(result.values())
    #datamap_stats = list(datamap_stats.values())

    return datamap_stats

def calc_confidence(df_probabilities):
    '''Calculates confidence by taking the mean for each instance probability at ground truth over all epochs'''
    x = np.array(df_probabilities['Probabilities'].tolist())
    confidence = np.mean(x.astype(float), axis=1)
    return confidence

def calc_variance(df_probabilities):
    '''Calculates variance by taking the standard deviation for each instance probability at ground truth over all epochs'''
    x = np.array(df_probabilities['Probabilities'].tolist())
    variance = np.std(x.astype(float), axis=1)
    return variance

def calc_correctness(df_correct_preds):
    '''Calculates correctness by measuring the number of times prediction == target'''
    x = np.array(df_correct_preds['Corrects'].tolist())
    boolean_array = (x=='True')
    sums = np.sum(boolean_array, axis=1)
    correctness = sums/len(df_correct_preds.loc[0]['Corrects'])
    return correctness


def remove_duplicates(id_probs, probs_corrects, col_name):
    predictions = pd.DataFrame(list(zip(id_probs.keys(), probs_corrects)),
    columns =['Question ID', col_name])
    predictions_duplicate = predictions[col_name].values
    predictions_remove_duplications = []
    for pred in predictions_duplicate:
        update = list(set(pred))
        if col_name != 'Predictions':
            if len(update) != 1:
                print(col_name)
                print(update)
                print("INVALID")
        predictions_remove_duplications.append(list(set(pred)))
    predictions[col_name] = predictions_remove_duplications
    return predictions

def calculate_datamap_metrics(datamap_stats, correctness_check=False):
    '''
    Returns dataframe with confidence, correctness and variability for datamap plotting

            Parameters:
                    datamap_stats (list): List of instance metadata for datamap 
                                          generation bucketed by epoch

            Returns:
                    df (DataFrame): Pandas dataframe with datamap metrics and question ids
    '''
    ids_probs = {}
    for example in datamap_stats[0]:
        ids_probs[example['Question ID']]=([],[],[],[],[],[]) # probabilities, corrects, question, target, prediction, image_id 

    for epoch_instances in datamap_stats:
        for example in epoch_instances:
            if example['Target'] == 'geese' and example['Prediction'] == 'goose': # special case in animal split
                correct = True
            elif example['Prediction'][-1] == 's' and example['Prediction'][-1] == example['Target']: # account for plurals in animal split
                correct = True
            elif example['Target'] == example['Prediction']:
                correct = True
            else:
                correct = False
            ids_probs[example['Question ID']][0].append(example['GT Probability'])
            ids_probs[example['Question ID']][1].append(correct)
            ids_probs[example['Question ID']][2].append(example['Question'])
            ids_probs[example['Question ID']][3].append(example['Target'])
            ids_probs[example['Question ID']][4].append(example['Prediction'])
            ids_probs[example['Question ID']][5].append(example['Image ID'])

    probs_corrects = np.array(list(ids_probs.values()))
    probabilities = pd.DataFrame(list(zip(ids_probs.keys(), probs_corrects[:, 0, :])),
                columns =['Question ID', 'Probabilities'])

    corrects = pd.DataFrame(list(zip(ids_probs.keys(), probs_corrects[:, 1, :])),
                columns =['Question ID', 'Corrects'])

    predictions = remove_duplicates(ids_probs, probs_corrects[:, 4, :], 'Predictions')
    questions = remove_duplicates(ids_probs, np.squeeze(probs_corrects[:, 2, :]), 'Question')
    targets = remove_duplicates(ids_probs, np.squeeze(probs_corrects[:, 3, :]), 'Target')
    image_ids = remove_duplicates(ids_probs, np.squeeze(probs_corrects[:, 5, :]), 'Image ID')

    confidence_score = calc_confidence(probabilities)
    variance_score = calc_variance(probabilities)
    correctness_score = calc_correctness(corrects)

    d= {"confidence": np.squeeze(confidence_score), "variability": np.squeeze(variance_score), "correctness":correctness_score, "question_id":probabilities['Question ID'],
        "Predictions": predictions['Predictions'], "Question":questions['Question'], "Target":targets['Target'], "Image ID":image_ids['Image ID']}
    df= pd.DataFrame(d)

    if correctness_check == True:
        low_correctness_high_confidence = df.loc[(df['correctness'] < 0.4) & (df['confidence'] > 0.8)]

        ids = low_correctness_high_confidence.index

        for id in ids:  
            x = probabilities.loc[id]
            y = corrects.loc[id]

            d_exp = {'probabilities':x, 'correctness': y}
            df_exp = pd.DataFrame(data=d)
            print(df_exp)

    return df

def plot_trainval_acc(base_path):
    '''
    Plots train/val accuracy scores 

            Parameters:
                    base_path (str): Path to model metadata

    '''
    with open(base_path + 'log.log') as fp:
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
    plt.savefig(base_path+'/training.png')

    plt.plot(xs_train, train_scores, label="Training")
    #plt.title("Training")
    plt.savefig(base_path+'/validation.png')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Correct Preds")

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
        plot.legend(ncol=1, bbox_to_anchor=(1.01, 0.5), loc='center left', fancybox=True, shadow=True)
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
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

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')

        plot2 = sns.countplot(x="correct.", data=dataframe, color='#86bf91', ax=ax3)
        ax3.xaxis.grid(True) # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('')

    fig.tight_layout()
    #filename = f'figures/{title}_{model}.pdf' if show_hist else f'figures/compact_{title}_{model}.pdf'
    #fig.savefig(filename, dpi=300)

def datamap_outliers(df, datamap_stats, confidence_threshold, variability_threshold, region):
    '''
    Extracts instances in datamap regions: hard, ambiguous, and easy

            Parameters:
                    df (DataFrame): Pandas dataframe with datamap metrics and question ids

                    datamap_stats (list): List of instance metadata for datamap 
                                          generation bucketed by epoch

                    confidence_threshold (float): threshold at which to segment the datamap (y axis) 

                    variability_threshold (float): threshold at which to segment the datamap (x axis) 

                    region (str): 'easy', 'hard', 'ambiguous', region of datamap 

            Returns:
                    outliers_metadata (list): List of metadata for outlier instances

                    img_ids (set): Image ids of outliers for extraction from dataset 
    '''

    if region == 'hard':
        # segment instances thresholded on confidence and variability to separate regions on datamap 
        outliers = df.loc[(df['confidence'] < confidence_threshold) & (df['variability'] < variability_threshold)]
    elif region == 'easy':
        outliers = df.loc[(df['confidence'] > confidence_threshold) & (df['variability'] < variability_threshold)]
    else:
        outliers = df.loc[(df['confidence'] > confidence_threshold) & (df['variability'] > variability_threshold)]
    ques_ids = outliers['question_id']

    outliers_metadata = []
    data_batch = datamap_stats[0]
    for id in ques_ids:
        for j in data_batch:
            if j['Question ID'] == id:
                outliers_metadata.append(j)
    img_ids = []
    for example in outliers_metadata:
        img_ids.append(example['Image ID'])

    img_ids_train = []
    img_ids_val = []

    for i in img_ids:
        if 'val' in i:
            img_ids_val.append(int(i[-6:]))
        else:
            img_ids_train.append(int(i[-6:]))
    return outliers_metadata, set(img_ids_train), set(img_ids_val)


def generate_datamap_images(df, base_path):
    image_ids = list(itertools.chain.from_iterable(df['Image ID']))
    image_ids_val = []
    image_ids_train = []
    for id in image_ids:
        if 'val' in id:
            image_ids_val.append(int(id[-6:]))
        else:
            image_ids_train.append(int(id[-6:]))    
    dataset_val = foz.load_zoo_dataset(
                    "coco-2014",
                    split='validation',
                    image_ids=image_ids_val,
                    dataset_dir=base_path+'datamap_imgs'
                )
    dataset_train = foz.load_zoo_dataset(
                    "coco-2014",
                    split='train',
                    image_ids=image_ids_train,
                    dataset_dir=base_path+'datamap_imgs'
                )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

base_path ='../snap/vqa/vqa_lxr955_animals_fromScratch_20epochs/'
datamap_stats = load_datamap_stats(base_path)
dataframe = calculate_datamap_metrics(datamap_stats)
#generate_datamap_stats(dataframe, base_path)
title='VQA-Animals From Scratch - 20 epochs'

# Subsample data to plot, so the plot is not too busy.
dataframe = dataframe.sample(n=25000 if dataframe.shape[0] > 25000 else len(dataframe))

# Normalize correctness to a value between 0 and 1.
dataframe['correct.'] = dataframe['correctness']

main_metric = 'variability'
other_metric = 'confidence'
hue_metric = 'correct.'

hue = hue_metric
num_hues = len(dataframe[hue].unique().tolist())
style = hue_metric if num_hues < 8 else None


# if not show_hist:
#     fig, axs = plt.subplots(1, 1, figsize=(8, 4))
#     ax0 = axs
# else:
#     fig = plt.figure(figsize=(16, 10), )
#     gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])

#     ax0 = fig.add_subplot(gs[0, :])


# ### Make the scatterplot.
# fig = px.scatter(dataframe, x=main_metric,
#                        y=other_metric, 
#                        color=hue,
#                        symbol=style,
#                        size_max=30)
# fig.show()

fig = go.Figure(data=[
    go.Scatter(
        x=dataframe[main_metric],
        y=dataframe[other_metric],
        mode="markers",
        marker=dict(
            colorscale='viridis',
            color=dataframe[hue],
            #size=df["MW"],
            colorbar={"title": "Correctness"},
            line={"color": "#444"},
            reversescale=True,
            sizeref=45,
            sizemode="diameter",
            opacity=0.8,
        ),
        title=title
    )
])

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

fig.update_layout(
    xaxis=dict(title=main_metric),
    yaxis=dict(title=other_metric),
    plot_bgcolor='rgba(255,255,255,0.1)'
)

#app = JupyterDash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)

def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = dataframe.iloc[num]
    #img_src = df_row['IMG_URL']
    #name = df_row['NAME']
    variability = df_row['variability']
    confidence = df_row['confidence']
    correctness = df_row['correctness']
    predictions = df_row['Predictions']
    question = df_row['Question']
    target = df_row['Target']
    
    #if len(desc) > 300: desc = desc[:100] + '...'
    image_id = str(df_row['Image ID'][0])
    if 'val' in image_id:
        split = 'validation'
    else:
        split = 'train'

    image = '../../fiftyone/coco-2014/'+split+'/data/'+image_id + '.jpg'
    encoded_image = base64.b64encode(open(image, 'rb').read())

    children = [
        html.Div(children=[
            #html.Img(src=img_src, style={"width": "100%"}),
            #html.H2(f"{name}", style={"color": "darkblue"}),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={"width": "100%"}),
            html.P(f"Question: {question}"),
            html.P(f"Target: {target}"),
            html.P(f"Predictions: {predictions}"),
            html.P(f"Variability: {variability}"),
            html.P(f"Confidence: {confidence}"),
            html.P(f"Correctness: {correctness}"),
        ],
        style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__=='__main__':
    app.run_server(debug=False, host='0.0.0.0', mode='inline')



