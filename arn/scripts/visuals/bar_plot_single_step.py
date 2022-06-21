"""Visualization with plotly for Experiment 1."""
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

def get_ocms_bar_plot(yaml_path):
    # TODO load in yaml file

    # TODO Get ocms from yaml file (as paths at leaves of dict tree, saved in place
    # of dict tree)
    #return [OrderedConfusionMatrices.load(path) for path in paths]
    # TODO Depth first : load ocms given each filepath at the leaf
    #while

    # TODO Get measures per ocm and format into dataframe for bar plot

    # TODO Create the bar plot
