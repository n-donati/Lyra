import random
import pandas as pd
import numpy as np

def generate_matrix(size, regions, local_density, distant_density):
    """
    Generate an adjacency matrix with specified parameters
    
    Args:
        size: Total number of neurons
        regions: Number of brain regions
        local_density: Probability of connection within same region (0-1)
        distant_density: Probability of connection with other regions (0-1)
    """
    # Create empty adjacency matrix
    matrix = np.zeros((size, size))
    
    # Calculate neurons per region
    nodes_per_region = size // regions
    remainder = size % regions
    
    # Create region assignments
    region_assignments = []
    current_pos = 0
    for i in range(regions):
        region_size = nodes_per_region + (1 if i < remainder else 0)
        region_assignments.extend([i] * region_size)
    
    # Generate connections
    for i in range(size):
        current_region = region_assignments[i]
        for j in range(i + 1, size):  # Upper triangle only
            target_region = region_assignments[j]
            
            # Determine connection probability
            if current_region == target_region:
                prob = local_density
            else:
                prob = distant_density
                
            # Create symmetric connection
            if random.random() < prob:
                matrix[i][j] = 1
                matrix[j][i] = 1  # Make it symmetric
    
    return matrix.tolist()  # Convert to Python list for JSON serialization

# Example usage:
if __name__ == "__main__":
    # Generate a larger test matrix
    test_matrix = generate_matrix(
        size=400,          # Default size
        regions=18,        # Default regions
        local_density=0.7, # Default local density
        distant_density=0.2 # Default distant density
    )
    
    # Print matrix size and sample
    print(f"Matrix size: {len(test_matrix)}x{len(test_matrix)}")
    print("Sample of first 5x5 elements:")
    for row in test_matrix[:5]:
        print([int(x) for x in row[:5]])