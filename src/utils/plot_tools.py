import numpy as np
from plotly import graph_objects as go
from numpy.typing import NDArray
from utils import HTML_FIGS

def plot_bubble_positions(
        bubble_positions: NDArray[np.float64],
        bubble_sizes: NDArray[np.float64],
        size_scale: float = 1.0,
        file_name: str = "layout") -> go.Figure:
    
    """
    Plot generated scene as a html file using plotly
    """

    fig = go.Figure(go.Scatter3d(
        x=bubble_positions[:, 0],
        y=bubble_positions[:, 1],
        z=bubble_positions[:, 2],
        mode='markers',
        marker = dict(
            size = bubble_sizes * size_scale,
            color = bubble_sizes,
            colorscale = 'jet',
            opacity = 0.6,
            showscale=True
        ),
        hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>z: %{z:.2f} μm<br>r: %{marker.color:.3f} μm',
    ))

    fig.update_layout(
            title=file_name,
            scene=dict(
                xaxis_title='x [μm]',
                yaxis_title='y [μm]',
                zaxis_title='z [μm]',
                aspectmode='data',
            ),
            margin=dict(l=100, r=100, b=100, t=100)
        )
    
    fig.write_html(HTML_FIGS / f"{file_name}.html")
    return fig