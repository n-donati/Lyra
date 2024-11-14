import random
import csv
import os
import numpy as np
from typing import List, Dict, Set, Tuple

def generate_hierarchical_network(num_nodes: int, num_clusters: int,
                                base_intra_density: float,
                                base_inter_density: float,
                                max_connections: int) -> List[List[str]]:
    """Generate network with hierarchical clustering and varying cluster strengths."""
    connections = [[] for _ in range(num_nodes)]
    
    # Create more balanced clusters
    min_cluster_size = max(3, num_nodes // (num_clusters * 2))  # Minimum size constraint
    max_cluster_size = num_nodes // (num_clusters // 2)  # Maximum size constraint
    
    # Initialize clusters with minimum sizes
    cluster_sizes = [min_cluster_size for _ in range(num_clusters)]
    remaining_nodes = num_nodes - (min_cluster_size * num_clusters)
    
    # Distribute remaining nodes more evenly
    while remaining_nodes > 0:
        for i in range(num_clusters):
            if remaining_nodes <= 0:
                break
            additional = min(
                random.randint(1, 3),  # Add 1-3 nodes at a time
                max_cluster_size - cluster_sizes[i],  # Don't exceed max size
                remaining_nodes  # Don't exceed available nodes
            )
            cluster_sizes[i] += additional
            remaining_nodes -= additional

    # Assign nodes to clusters
    clusters: Dict[int, List[int]] = {}
    current_node = 0
    cluster_strengths: Dict[int, float] = {}
    
    for i, size in enumerate(cluster_sizes):
        clusters[i] = list(range(current_node, current_node + size))
        # Assign random strength to each cluster
        cluster_strengths[i] = random.uniform(0.5, 1.5)
        current_node += size

    def add_connection(node1: int, node2: int, force: bool = False):
        if node2 not in connections[node1]:
            if force or len(connections[node1]) < max_connections:
                connections[node1].append(node2)
                connections[node2].append(node1)
                return True
        return False

    # Generate core connections within clusters
    for cluster_id, nodes in clusters.items():
        strength = cluster_strengths[cluster_id]
        
        # Create a strongly connected core in each cluster
        core_size = max(2, len(nodes) // 3)
        core_nodes = nodes[:core_size]
        
        # Ensure core is well connected
        for i in range(len(core_nodes)):
            for j in range(i + 1, len(core_nodes)):
                if random.random() < 0.8 * strength:
                    add_connection(core_nodes[i], core_nodes[j], force=True)

        # Connect remaining nodes with varying probability
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                prob = base_intra_density * strength
                if i < core_size:
                    prob *= 1.5  # Increase probability for core nodes
                if random.random() < prob:
                    add_connection(nodes[i], nodes[j])

    # Generate inter-cluster connections based on cluster strengths
    cluster_pairs = [(i, j) for i in range(len(clusters)) for j in range(i + 1, len(clusters))]
    for c1, c2 in cluster_pairs:
        # Calculate inter-cluster connection probability based on cluster strengths
        strength_factor = min(cluster_strengths[c1], cluster_strengths[c2])
        prob = base_inter_density * strength_factor
        
        # Connect only a few nodes between clusters
        sample_size1 = max(1, len(clusters[c1]) // 4)
        sample_size2 = max(1, len(clusters[c2]) // 4)
        
        for node1 in random.sample(clusters[c1], sample_size1):
            for node2 in random.sample(clusters[c2], sample_size2):
                if random.random() < prob:
                    add_connection(node1, node2)

    # Ensure no isolated nodes
    def ensure_connectivity():
        for node in range(num_nodes):
            if not connections[node]:
                # Find closest node in same cluster or neighboring cluster
                for cluster_id, nodes in clusters.items():
                    if node in nodes:
                        # Try connecting to nodes in same cluster first
                        possible_connections = [n for n in nodes if n != node]
                        if possible_connections:
                            target = random.choice(possible_connections)
                            add_connection(node, target, force=True)
                            break
                        
                        # If still isolated, connect to any node with space
                        if not connections[node]:
                            for other_node in range(num_nodes):
                                if other_node != node and len(connections[other_node]) < max_connections:
                                    add_connection(node, other_node, force=True)
                                    break

    # Add this call before formatting the data
    ensure_connectivity()

    # Format connections for CSV
    formatted_data = [["node", "connections"]]
    for i, conn in enumerate(connections, 1):
        formatted_data.append([str(i), ",".join(map(str, sorted([x + 1 for x in conn])))])

    return formatted_data

def save_network(data: List[List[str]], filename: str):
    """Save network data to CSV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

if __name__ == "__main__":
    # Configurable parameters
    NUM_NODES = 124
    NUM_CLUSTERS = 12
    BASE_INTRA_DENSITY = 0.1
    BASE_INTER_DENSITY = 0.05
    MAX_CONNECTIONS = 4
    
    network_data = generate_hierarchical_network(
        NUM_NODES,
        NUM_CLUSTERS,
        BASE_INTRA_DENSITY,
        BASE_INTER_DENSITY,
        MAX_CONNECTIONS
    )
    save_network(network_data, "data/network.csv")
    print(f"Network data generated with {NUM_NODES} nodes in {NUM_CLUSTERS} clusters")
