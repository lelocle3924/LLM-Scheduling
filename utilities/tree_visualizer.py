import os
import matplotlib.pyplot as plt
import networkx as nx


def _get_tree_positions(root_node):
    """Recursively calculates X, Y coordinates for a tree layout."""
    positions = {}

    def traverse(node, depth, min_x, max_x):
        # Assign coordinates
        x = (min_x + max_x) / 2
        y = -depth  # Negative so the tree grows downwards
        positions[id(node)] = (x, y)

        # Recurse for children
        children = list(node.children.values())
        if children:
            width = (max_x - min_x) / len(children)
            for i, child in enumerate(children):
                child_min_x = min_x + i * width
                child_max_x = min_x + (i + 1) * width
                traverse(child, depth + 1, child_min_x, child_max_x)

    # Start traversal with an arbitrary width span
    traverse(root_node, depth=0, min_x=0, max_x=100)
    return positions


def draw_mcts_tree(root_node, selected_leaf, output_path):
    """
    Renders the MCTS tree to a PNG using NetworkX and Matplotlib.
    Selected nodes in the current loop are saturated blue, others are pale blue.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Trace the path of the current MCTS loop (from leaf to root)
    selected_path = set()
    curr = selected_leaf
    while curr is not None:
        selected_path.add(id(curr))
        curr = curr.parent

    G = nx.DiGraph()
    labels = {}
    node_colors = []

    # Build graph nodes and edges
    def build_graph(node):
        node_id = id(node)
        is_selected = node_id in selected_path

        # Styling
        node_colors.append("#4169E1" if is_selected else "#D0E8F2")

        # Labeling
        if node.parent is None:
            labels[node_id] = f"ROOT\nN={node.visits}\nQ={node.q_value:.2f}"
        else:
            act = node.action
            labels[node_id] = f"J{act['job']}O{act['op']}->M{act['machine']}\nN={node.visits}\nQ={node.q_value:.2f}"

        G.add_node(node_id)

        for child_node in node.children.values():
            child_id = id(child_node)
            G.add_edge(node_id, child_id)
            build_graph(child_node)

    build_graph(root_node)
    pos = _get_tree_positions(root_node)

    # Calculate dynamic figure size based on tree width/depth
    depths = [-y for x, y in pos.values()]
    max_depth = max(depths) if depths else 1
    leaf_count = len([n for n in G.nodes() if G.out_degree(n) == 0])
    fig_width = max(10, leaf_count * 1.5)
    fig_height = max(6, max_depth * 2)

    # Plotting
    plt.figure(figsize=(fig_width, fig_height))

    # Draw edges: highlight the selected path
    edge_colors = [
        "#4169E1" if (u in selected_path and v in selected_path) else "#A9A9A9"
        for u, v in G.edges()
    ]
    edge_widths = [
        2.5 if (u in selected_path and v in selected_path) else 1.0
        for u, v in G.edges()
    ]

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=False)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_shape="s",  # square/box
        node_size=4000,
        edgecolors="black",
        linewidths=1.0
    )

    # Draw labels
    font_colors = {n: "white" if n in selected_path else "black" for n in G.nodes()}
    for node_id, text in labels.items():
        nx.draw_networkx_labels(
            G, pos,
            labels={node_id: text},
            font_color=font_colors[node_id],
            font_size=9,
            font_weight="bold" if node_id in selected_path else "normal"
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path + ".png", dpi=150, bbox_inches="tight")
    plt.close()
