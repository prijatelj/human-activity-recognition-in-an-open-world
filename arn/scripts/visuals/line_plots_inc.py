"""Visualization of experiment 2."""
from functools import partial
import glob
import os
import re

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.express.colors import qualitative as qual_colors
from plotly.validators.scatter.marker import SymbolValidator
import yaml

from exputils.data import OrderedConfusionMatrices
from docstr.cli.cli import docstr_cap


def line_plot(
    df,
    uid_col='Feature Repr.',
    k6_start_line=1,
    k7_start_line=5,
    steps='Step',
    measure='Matthews Correlation Coefficient',
    line_colors='Vivid',
    line_dashes=None,
    font_size=20,
    x_range=None,
    x_dtick=None,
    y_range=None,
    y_dtick=None,
    #y_minor=None,
    legend=None,
    symbols=None,
    margin=None,
    width=None,
    height=None,
):
    if isinstance(line_colors, str):
        line_colors = list(getattr(qual_colors, line_colors))
    if symbols is None:
        symbols = SymbolValidator().values
    #if y_minor is None:
    #    y_minor = dict(
    #        showgrid=True,
    #        tickcolor='black',
    #    )
    if y_range is None:
        y_range = [0.0, 1.0]
    if legend is None:
        legend = dict(
            yanchor='bottom',
            y=0.01,
            xanchor='right',
            x=0.99,
        )
    if margin is None:
        margin = dict(l=0, r=0, t=20, b=0, pad=0)

    fig = go.Figure()
    if k7_start_line:
        fig.add_trace(go.Scatter(
            x=[k6_start_line, k6_start_line],
            y=[0, 1],
            mode='lines',
            name='Kinetics-600 Start',
            line=dict(
                #width=2,
                color='rgba(128, 128, 128, 1.0)',
                dash='dot',
            ),
        ))
    if k7_start_line:
        fig.add_trace(go.Scatter(
            x=[k7_start_line, k7_start_line],
            y=[0, 1],
            mode='lines',
            name='Kinetics-700-2020 Start',
            line=dict(
                #width=2,
                color='rgba(128, 128, 128, 1.0)',
                dash='dash',
            ),
        ))

    # Loop thru and add lines
    for i, val in enumerate(df[uid_col].unique()):
        view = df[df[uid_col] == val]
        fig.add_trace(go.Scatter(
            x=view[steps],
            y=view[measure],
            mode='lines+markers',
            name=val,
            marker_color=line_colors[i],
            marker_symbol=symbols[i],
            #error_y=None if measure_error is None else measure_error[key],
            line=None if line_dashes is None else dict(dash=line_dashes[i]),
        ))

    fig.update_layout(
        template='simple_white',
        font=dict(
            size=font_size,
        ),
        legend=legend,
        xaxis=dict(
            #zeroline=True,
            #zerolinecolor='black',
            tickwidth=2,
            showgrid=True,
            showline=True,
            title='Increments',
            range=x_range,
            dtick=x_dtick,
        ),
        yaxis=dict(
            #zeroline=True,
            #zerolinecolor='black',
            tickwidth=2,
            showgrid=True,
            showline=True,
            title=measure,
            range=y_range,
            dtick=y_dtick,
            #minor_ticks=y_minor, # Why does this not exist?
        ),
        margin=margin,
        width=width,
        height=height,
    )
    return fig


def square_mcc_exp2(df):
    return line_plot(
        df[~df['Pre-feedback']],
        x_range=[0,10],
        y_range=[0.62, 0.73],
        x_dtick=1,
        symbols=[
            'star-triangle-up',
            'star-triangle-down',

            'star-diamond',
            'star-square',

            'x',
            'cross',

            'square',
            'diamond',
        ],
        uid_col='uid',
        legend=dict(
            yanchor='bottom',
            y=0.01,
            xanchor='left',
            x=0.52,
        ),
        line_dashes=['solid', 'dash'] * 4,
        margin=dict(l=0, r=20, t=0, b=0, pad=0),
        width=800,
        height=800,
    )


def wide_mcc_exp2(df):
    # Validation specifically.
    return line_plot(
        df[~df['Pre-feedback']],
        x_range=[0,10],
        y_range=[0.49, 0.75],
        x_dtick=1,
        symbols=[
            'star-triangle-up',
            'star-triangle-down',

            'star-diamond',
            'star-square',

            'x',
            'cross',

            'square',
            'diamond',
        ],
        uid_col='uid',
        legend=dict(
            yanchor='bottom',
            y=0.04,
            xanchor='right',
            x=0.3,
        ),
        line_dashes=['solid', 'dash'] * 4,
        margin=dict(l=0, r=20, t=0, b=0, pad=0),
        width=1600,
        height=400,
    )


def wide_nmi_exp2_legend_on_side(df):
    return line_plot(
        df,
        #measure='Accuracy',
        measure='Normalized Mutual Information',
        x_range=[0,10.6],
        y_range=[-0.1, 1.0],
        x_dtick=1,
        symbols=[
            'star-triangle-up',
            'star-triangle-down',

            'star-diamond',
            'star-square',

            'x', #'x-thin',
            'cross', #'cross-thin',

            'square',
            'diamond',
        ],
        uid_col='uid',
        legend=dict(
            yanchor='bottom',
            y=0.1,
            xanchor='left',
            x=1.01,
        ),
        line_dashes=['solid', 'dash'] * 4,
        margin=dict(l=0, r=20, t=0, b=0, pad=0),
        width=1600,
        height=400,
    )


if __name__ == '__main__':
    #df = load_incremental_ocms_df(...)
    df = pd.read_csv('/media/har//results/kowl/exp2_val_set_possibly_overlapping_training.csv')

    fig = line_plot(
        df[~df['Pre-feedback']],
        x_range=[0,10],
        x_dtick=1,
        symbols=['circle', 'diamond'],
        uid_col='uid',
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0, pad=0),
        width=1600,
        height=390,
    )
    plotly.io.write_image(fig, 'output_path_of_nifty_figure.pdf', format='pdf')
