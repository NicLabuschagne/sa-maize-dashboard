import pandas as pd
import plotly.graph_objects as go


def line(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines"))
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20), height=380)
    return fig
