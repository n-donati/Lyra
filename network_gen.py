# First, let's generate the CSV using Python
import random
import pandas as pd
import numpy as np

# Generate neurons with regional connectivity
def generate_neural_network():
    neurons = 250
    nodes_per_region = neurons // 18
    remainder = neurons % 18
    
    # Create 18 brain regions with roughly equal distribution
    regions = {}
    current_start = 1
    
    region_names = [
        'frontal_sup', 'frontal_inf', 'prefrontal',
        'temporal_sup', 'temporal_inf', 'temporal_med',
        'parietal_sup', 'parietal_inf', 'parietal_med',
        'occipital_sup', 'occipital_inf', 'occipital_lat',
        'motor_primary', 'motor_supplementary', 'sensory_primary',
        'limbic', 'subcortical', 'cerebellar'
    ]
    
    for i, name in enumerate(region_names):
        size = nodes_per_region + (1 if i < remainder else 0)
        regions[name] = list(range(current_start, current_start + size))
        current_start += size
    
    connections = []
    for neuron in range(1, neurons + 1):
        # Determine region of current neuron
        current_region = [r for r, n in regions.items() if neuron in n][0]
        
        # Higher probability to connect within same region
        same_region_neurons = regions[current_region]
        other_neurons = [n for n in range(1, neurons + 1) if n not in same_region_neurons]
        
        # Create connections with bias towards same region
        local_connections = random.sample(same_region_neurons, random.randint(1, 4))
        distant_connections = random.sample(other_neurons, random.randint(2, 3))
        all_connections = local_connections + distant_connections
        
        connections.append({
            'node': neuron,
            'connections': ','.join(map(str, [c for c in all_connections if c != neuron]))
        })
    
    return pd.DataFrame(connections)

# Generate and save the network
network = generate_neural_network()
network.to_csv('data/network.csv', index=False)