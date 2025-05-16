import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import Voronoi
from shapely.geometry import Polygon,Point
from adjustText import adjust_text
import os
import shutil

def plot_brillouin_zone(a1,
                         a2,
                         user_points_reciprocal=None,
                         n_range=2,
                         offset_mag=0.1,
                         figsize=(6, 6),
                         zone_color='lightblue',
                         zone_alpha=0.5,
                         label_user_points=True):
    """
    Compute and plot the 1st Brillouin Zone for a 2D lattice defined by real-space vectors a1 and a2.

    Parameters
    ----------
    a1, a2 : array-like, shape (2,)
        Real-space lattice vectors.
    user_points_reciprocal : array-like, shape (M, 2), optional
        Points specified in reciprocal-basis coordinates to plot.
    n_range : int, default=2
        Range for generating reciprocal lattice points: [-n_range, n_range].
    offset_mag : float, default=0.1
        Magnitude of perpendicular offset when labeling midpoints.
    figsize : tuple, default=(6,6)
        Figure size for the plot.
    zone_color : str, default='lightblue'
        Fill color for the Brillouin Zone polygon.
    zone_alpha : float, default=0.5
        Alpha transparency for the Brillouin Zone fill.
    label_user_points : bool, default=True
        If True, annotate the user-specified reciprocal points; if False, plot markers only.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects with the Brillouin Zone plot.
    """
    # Convert inputs
    a1 = np.array(a1, dtype=float)
    a2 = np.array(a2, dtype=float)

    # Compute unit cell area and reciprocal vectors
    area = np.abs(a1[0]*a2[1] - a1[1]*a2[0])
    b1 = 2 * np.pi * np.array([-a2[1], a2[0]]) / area
    b2 = 2 * np.pi * np.array([ a1[1], -a1[0]]) / area
    B = np.column_stack((b1, b2))

    # Generate reciprocal lattice points
    rec_points = []
    for i in range(-n_range, n_range+1):
        for j in range(-n_range, n_range+1):
            rec_points.append(i * b1 + j * b2)
    rec_points = np.array(rec_points)

    # Voronoi diagram and region for origin
    vor = Voronoi(rec_points)
    region_index = next(
        (r for (pt, r) in zip(rec_points, vor.point_region) if np.allclose(pt, 0)),
        None
    )
    if region_index is None:
        raise ValueError("Origin not found in reciprocal points.")

    # Extract, order, and close vertices
    verts = vor.vertices[vor.regions[region_index]]
    poly = Polygon(verts)
    ordered = np.array(poly.exterior.coords)
    n_unique = len(ordered) - 1

    # Begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    # Brillouin Zone
    ax.fill(ordered[:,0], ordered[:,1], color=zone_color, alpha=zone_alpha)
    ax.plot(ordered[:,0], ordered[:,1], 'b-')
    ax.plot(0, 0, 'ro')

    # Plot b1, b2
    ax.arrow(0, 0, b1[0], b1[1], head_width=0.15, head_length=0.15,
             fc='green', ec='green', length_includes_head=True)
    ax.text(b1[0]*1.1, b1[1]*1.1, 'b1', color='green')
    ax.arrow(0, 0, b2[0], b2[1], head_width=0.15, head_length=0.15,
             fc='purple', ec='purple', length_includes_head=True)
    ax.text(b2[0]*1.1, b2[1]*1.1, 'b2', color='purple')

    # Label vertices in reciprocal basis
    for v in ordered[:n_unique]:
        c = np.linalg.solve(B, v)
        ax.text(v[0], v[1], f"({c[0]:.2f}, {c[1]:.2f})",
                color='red', ha='center', va='center')

    # Midpoint labels
    for i in range(n_unique):
        v1, v2 = ordered[i], ordered[(i+1)%n_unique]
        mid = (v1+v2)/2
        edge = v2 - v1
        perp = np.array([-edge[1], edge[0]])
        perp /= np.linalg.norm(perp)
        offset = perp * offset_mag
        cmid = np.linalg.solve(B, mid)
        ax.plot(mid[0], mid[1], 'ko')
        ax.text(mid[0]+offset[0], mid[1]+offset[1],
                f"({cmid[0]:.2f}, {cmid[1]:.2f})",
                color='black', ha='center', va='center')

    # User points (always plot markers)
    if user_points_reciprocal is not None:
        up = np.array(user_points_reciprocal)
        cart = up.dot(B.T)
        # plot markers
        for ucart in cart:
            ax.plot(ucart[0], ucart[1], 'm*', markersize=5)
        # optionally add labels
        if label_user_points:
            texts = []
            for (urec, ucart) in zip(up, cart):
                tt = ax.text(ucart[0], ucart[1], f"({urec[0]:.3f}, {urec[1]:.3f})", color='magenta')
                texts.append(tt)
            adjust_text(
                texts,
                ax=ax,
                only_move={'points':'y', 'texts':'y'},
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )

    ax.set_title("2D Brillouin Zone")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.axis('equal')
    ax.legend(["Zone boundary", "Origin", "b1, b2", "Vertices", "Midpoints", "User points"],
              loc='upper right', fontsize='small')
    plt.show()
    return fig, ax
