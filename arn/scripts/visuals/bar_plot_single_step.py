"""Visualization with plotly for Experiment 1."""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative as qual_colors
import yaml

from exputils.data import OrderedConfusionMatrices

from arn.scripts.visuals.load_results import load_inplace_results_tree


def bar_group(
    df,
    groupby='columns',
    title=None,
    measure_range=None,
    measure_title=None,
    measure_title_font_size=16,
    measure_tick_font_size=14,
    measure_error=None,
    xaxis_tickfont_size=14,
    x_title=None,
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1, # gap between bars of the same location coordinate.
    marker_colors=None,
    plot_bgcolor=None,
):
    """Plot the given measures as bars, grouping either by columns or indices.
    """
    fig = go.Figure()

    if groupby == 'columns':
        values = df.values.T
        x = df.index
        names = df.columns
    else:
        values = df.values
        x = df.columns
        names = df.index

    # Add a bar for every measure
    for key, measures in enumerate(values):
        fig.add_trace(go.Bar(
            x=x,
            y=measures,
            name=names[key],
            marker_color=None if marker_colors is None else marker_colors[key],
            error_y=None if measure_error is None else measure_error[key],
        ))

    # Update general layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickfont_size=xaxis_tickfont_size,
            title=x_title,
        ),
        yaxis=dict(
            title=measure_title,
            titlefont_size=measure_title_font_size,
            tickfont_size=measure_tick_font_size,
            range=measure_range,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=bargap,
        bargroupgap=bargroupgap,
        plot_bgcolor=plot_bgcolor,
    )

    return fig

#fig = bar_group(
#    acc,
#    title="Feature Representations's Augmented Video Performance",
#    marker_colors=colors,
#    measure_range=[0,1],
#    plot_bgcolor='rgb(200, 200, 200)',
#    measure_title='Accuracy [0, 1]'
#)


def bar_multigroup(
    df,
    col_group_order,
    measure_col,
    marker_colors=None,
    measure_error=None,
    split_kwargs=None,
    split_col='Data Split',
    classifier_col='Classifier',
    frepr_col='Feature Repr.',
):
    """Create a bar graph comparing multiple models' performance measures where
    the models are grouped by feature representation, classifier, and data
    split. The data splits' values are opt. overlaid. The other groups are
    side-by-side.
    """
    if groupby == 'columns':
        values = df.values.T
        x = df.index
        names = df.columns
    else:
        values = df.values
        x = df.columns
        names = df.index

    bar_colors = [qual_colors.Pastel2[-1]] + list(
        np.array(qual_colors.Pastel1)[[1, 6, 3, 2, 4, 0]]
    )

    if split_kwargs is None:
        split_kwargs = {
            'train': {
                'opacity': 0.5,
                'line': {
                    'width': 2,
                    'color': 'white',
                },
                'pattern': {
                    'shape': '/',
                    'fillmode': 'replace',
                    'solidity': 0.7,
                },
            },
            'val': {
                'opacity': 0.5,
                'line': {
                    'width': 2,
                    'color': 'white',
                },
                'pattern': {
                    'shape': 'x',
                    'fillmode': 'replace',
                    'solidity': 0.7,
                },
            },
            'test': {
                'line': {
                    'width': 2,
                    'color': 'white',
                },
            }
        }

    freprs = df[frepr_col].unique()
    classifiers = df[classifier_col].unique()

    fig = make_subplots(
        rows=1,
        cols=len(freprs),
        shared_yaxes=True,
        specs=[[{}, {}], [{}, {}]]
    )

    # TODO first, make stacked bar graph, stacking splits and comparing
    # classifiers for a single frepr.

    # TODO For each frepr + classifier pair, create the stacked splits
    #   Each split has its own fill/pattern shape and line style, overlaid each
    #   other (or side by side, optional).
    #   Each classifier has its own color.
    frepr_figs = []
    for i, frepr in enumerate(freprs):
        for split in ['train', 'val', 'test']:
            fig.add_trace(
                go.Bar(
                    name=split,
                    x=classifiers,
                    y=df[df["Data Split"]==split]["Mathew's Correlation Coefficient"],
                    marker=split_kwargs[split],
                    marker_color=bar_colors[:len(classifiers)],
                    textposition='inside',
                    texttemplate='%{y:.4f}',
                    textfont_color='black',
                    #uid=f'{frepr}_{}',
                ),
                row=0,
                col=i,
            )
            fig.update_layout(barmode='overlay', template='simple_white')



    # TODO Join the stacked splits of each classifier for a frepr.
    #   White space sep between

    # TODO Join the frepr plots together into one. More ws separating freprs.


def get_ocms_bar_plot_df(yaml_path):
    # Load in yaml config file
    with open(yaml_path) as openf:
        config = yaml.load(openf, Loader=yaml.CLoader)

    root_dir = config.pop('root_dir', '')

    # Load ocms given each filepath at the leaf and store in-place
    load_inplace_results_tree(
        config['ocms'],
        root_dir,
        get_ocm=True,
        leaf_is_dir=False,
    )

    # Get measures per ocm and format into dataframe for bar plot
    df = pd.DataFrame(
        [],
        columns=['F.Repr.', 'Classifier', 'Data Split']
            + list(config['measures'].keys()),
    )
    for k1, v1 in config['ocms'].items():
        for k2, v2 in v1.items():
            for k3, ocm in v2.items():
                measures = []
                cm = ocm.get_conf_mat()
                for m_attr in config['measures'].values():
                    if isinstance(m_attr, dict):
                        key, val = next(iter(m_attr.items()))
                        if isinstance(val, dict):
                            if key == 'accuracy':
                                # top-k accuracy
                                measure = getattr(ocm, 'accuracy')(**val)
                            else:
                                measure = getattr(cm, key)(**val)
                        else: # probs never used
                            measure = getattr(cm, key)(val)
                    else:
                        measure = getattr(cm, m_attr)()
                    measures.append(measure)
                df = df.append(pd.Series(
                    [k1, k2, k3] + measures,
                    index=df.columns,
                ), ignore_index=True)
    return df
