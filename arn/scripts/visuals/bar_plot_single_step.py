"""Visualization with plotly for Experiment 1."""
import os

import pandas as pd
import plotly.graph_objects as go
import yaml

from exputils.data import OrderedConfusionMatrices


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

def load_ocm_tree_inplace(tree, root_dir=''):
    stack = [(tree, k, v) for k, v in tree.items()]
    while stack:
        ptr, key, value = stack.pop()
        if isinstance(value, dict):
            for k, v in value.items():
                stack.append((value, k, v))
        else:
            ptr[key] = OrderedConfusionMatrices.load(
                os.path.join(root_dir, value)
            )

def get_ocms_bar_plot(yaml_path):
    # Load in yaml config file
    with open(yaml_path) as openf:
        config = yaml.load(openf, Loader=yaml.CLoader)

    root_dir = config.pop('root_dir', '')

    # Load ocms given each filepath at the leaf and store in-place
    load_ocm_tree_inplace(config['ocms'], root_dir)

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

    # TODO Create the bar plot
