import pandas as pd
import numpy as np
import plotly.graph_objects as go

def eventstudyplot(input, big_title='Event Study Plot (With 95% CIs)', path_output='', name_output='eventstudyplot'):
    print('\nTakes output from est_ functions to plot event study estimates and their CIs')
    print('Requires 3 columns: parameter, lower, and upper; indexed to relative time')
    print('Output = plotly graph objects figure')
    d = input.copy()
    fig = go.Figure()
    # Point estimate
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d['parameter'],
            name='Coefficients on Lead / Lags of Treatment',
            mode='lines',
            line=dict(color='black', width=3)
        )
    )
    # Lower bound
    if len(d[~d['lower'].isna()]) > 0:
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d['lower'],
                name='Lower Confidence Bound',
                mode='lines',
                line=dict(color='black', width=1, dash='dash')
            )
        )
    # Upper bound
    if len(d[~d['upper'].isna()]) > 0:
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d['upper'],
                name='Upper Confidence Bound',
                mode='lines',
                line=dict(color='black', width=1, dash='dash')
            )
        )
    # Overall layout
    fig.update_layout(
        title=big_title,
        plot_bgcolor='white',
        font=dict(color='black')
    )
    # Save output
    fig.write_html(path_output + name_output + '.html')
    fig.write_image(path_output + name_output + '.png', height=768, width=1366)
    # fig.show()
    return fig
