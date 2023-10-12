
# networkx extension to draw edge labels on curved edges. see 
# https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def alpha_blend_with_background(foreground_hex, alpha, background_hex="#FFFFFF"):
    # Convert hex to RGB
    fg_rgb = [int(foreground_hex[i:i+2], 16) for i in (1, 3, 5)]
    bg_rgb = [int(background_hex[i:i+2], 16) for i in (1, 3, 5)]

    # Blend with background
    blended_rgb = [
        int((1 - alpha) * bg + alpha * fg)
        for fg, bg in zip(fg_rgb, bg_rgb)
    ]

    # Convert blended RGB back to hex
    blended_hex = "#{:02x}{:02x}{:02x}".format(*blended_rgb)

    return blended_hex

def draw_transition_matrix(transition_matrix, ax, arc_rad=0.2, layout_fun=nx.circular_layout, node_colors=None, node_color_alpha=0.2):

    # Get the list of states (nodes)
    states = transition_matrix.index.tolist()

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(states)

    # Add directed edges to the graph based on the transition matrix
    for from_state in states:
        for to_state in states:
            count = transition_matrix.at[from_state, to_state]
            if float(count) > 0:
                G.add_edge(from_state, to_state, weight=count)

    # Define the layout for the graph
    layout = layout_fun(G)

    # Define node colors

    node_edge_colors = ['#000000'] * len(G.nodes()) if node_colors is None else node_colors
    color_map = dict(zip(G.nodes(), node_edge_colors))
    
    node_colors = [alpha_blend_with_background(color, alpha=node_color_alpha) for color in node_edge_colors]

    nx.draw_networkx_nodes(G, layout, ax=ax, node_color=node_colors, edgecolors=node_edge_colors, node_size=500)
    nx.draw_networkx_labels(G, layout,  ax=ax)

    # Draw edges
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    straight_edge_colors = [color_map[edge[1]] for edge in straight_edges]
    nx.draw_networkx_edges(G, layout, ax=ax, edgelist=straight_edges, edge_color=straight_edge_colors, arrowsize=20)
    curved_edge_colors = [color_map[edge[1]] for edge in curved_edges]
    nx.draw_networkx_edges(G, layout, ax=ax, edgelist=curved_edges, edge_color=curved_edge_colors, connectionstyle=f'arc3, rad = {arc_rad}', arrowsize=20)

    # draw edges
    edge_weights = nx.get_edge_attributes(G,'weight')

    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    my_draw_networkx_edge_labels(G, layout, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad, )
    nx.draw_networkx_edge_labels(G, layout, ax=ax, edge_labels=straight_edge_labels,rotate=False)
    ax.margins(x=0.1, y=0.1)  # Adds 10% padding in both x and y directions


def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items