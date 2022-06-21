"""Visualization of experiment 2."""

# TODO Line Plot with error bars, horizontal axis is time-step

def line_plot_group(
    df,
    groupby='columns',
    title=None,
    measure_range=None,
    measure_title=None,
    measure_title_font_size=16,
    measure_tick_font_size=14,
    measure_nticks=5,
    measure_error=None,
    xaxis_tickfont_size=14,
    x_title=None,
    marker_colors=None,
    plot_bgcolor=None,
):
    """Line plot comparison of measures."""
    fig = go.Figure()

    if groupby == 'columns':
        values = df.values.T
        x = df.index
        names = df.columns
    else:
        values = df.values
        x = df.columns
        names = df.index

    for key, measures in enumerate(values):
        fig.add_trace(go.Scatter(
            x=x,
            y=measures,
            mode='lines+markers',
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
            showgrid=True,
            showline=True,
        ),
        yaxis=dict(
            title=measure_title,
            titlefont_size=measure_title_font_size,
            tickfont_size=measure_tick_font_size,
            range=measure_range,
            showgrid=True,
            showline=True,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        plot_bgcolor=plot_bgcolor,
    )

    return fig
