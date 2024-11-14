import itertools
import random

import numpy as np

skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
             [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1],
             [0, 2], [1, 3], [2, 4], [11, 12]]


def create_adjacency_matrix(skeletons):
    max_node = max(max(pair) for pair in skeletons)
    size = max_node + 1
    adj_matrix = np.zeros((size, size), dtype=int)

    for u, v in skeletons:
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1

    return adj_matrix


def find_connected_subgraphs(adj_matrix, n):
    num_nodes = len(adj_matrix)
    all_connected_subgraphs = []

    # Step 1: Find all combinations of nodes of size n
    for nodes in itertools.combinations(range(num_nodes), n):
        # Step 2: Check if the subset of nodes is connected
        if is_connected(adj_matrix, nodes):
            all_connected_subgraphs.append(nodes)

    return all_connected_subgraphs


def is_connected(adj_matrix, nodes):
    visited = set()
    # Step 2.1: Start DFS from the first node in the subset
    stack = [nodes[0]]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            # Add neighbors in the subset to the stack
            for neighbor in nodes:
                if adj_matrix[node][neighbor] == 1 and neighbor not in visited:
                    stack.append(neighbor)

    # Step 2.2: Check if we've visited all nodes in the subset
    return len(visited) == len(nodes)


adj_matrix = create_adjacency_matrix(skeletons)
print("Adjacency Matrix:")
print(adj_matrix)

n = 5  # Number of connected subgraph nodes
connected_subgraphs = find_connected_subgraphs(adj_matrix, n)
print("Connected Subgraphs:")
for subgraph in connected_subgraphs:
    print(subgraph)
print(len(connected_subgraphs))


def calculate_node_scores(adj_matrix):
    # Step 1: Calculate degrees of each node
    degrees = np.sum(adj_matrix, axis=1)
    max_degree = max(degrees)

    # Step 2: Assign scores inversely proportional to the degree
    scores = {node: max_degree - degree + 1 for node, degree in enumerate(degrees)}
    return scores


def score_subgraphs(subgraphs, scores):
    # Step 3: Calculate the score for each subgraph based on node scores
    subgraph_scores = []
    for subgraph in subgraphs:
        subgraph_score = sum(scores[node] for node in subgraph)
        subgraph_scores.append((subgraph, subgraph_score))

    # Sort subgraphs by score in descending order
    subgraph_scores.sort(key=lambda x: x[1], reverse=True)
    return subgraph_scores


def select_top_n_subgraphs(subgraphs, adj_matrix, n, type=2):
    scores = calculate_node_scores(adj_matrix)  # The subgraph with high scores and small
    scored_subgraphs = score_subgraphs(subgraphs, scores)

    if type == 1:  # Strategy 1: Maximum degree and
        scored_subgraphs.sort(key=lambda x: sum(scores[node] for node in x[0]))
    elif type == 2:  # Strategy 2: Minimum degree sum
        scored_subgraphs.sort(key=lambda x: sum(scores[node] for node in x[0]), reverse=True)
    elif type == 3:  # Strategy 3: Random Selection
        top_n_subgraphs = random.sample(scored_subgraphs, min(n, len(scored_subgraphs)))
        return [subgraph for subgraph, score in top_n_subgraphs]
    else:
        raise ValueError("Invalid type. Please use 1 (min degree sum), 2 (max degree sum), or 3 (random).")

    top_n_subgraphs = [subgraph for subgraph, score in scored_subgraphs[:n]]
    return top_n_subgraphs


print(select_top_n_subgraphs(connected_subgraphs, adj_matrix, 17, type=3))
