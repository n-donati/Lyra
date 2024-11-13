from django.shortcuts import render
from django.http import JsonResponse
import networkx as nx
import pandas as pd
from functools import lru_cache
import numpy as np
from sklearn.cluster import KMeans

@lru_cache(maxsize=1)
def load_network():
    G = nx.Graph()
    edges = []
    try:
        df = pd.read_csv('data/network.csv')
        for _, row in df.iterrows():
            edges.extend([(row['node'], target.strip()) 
                         for target in row['connections'].split(',')])
        G.add_edges_from(edges)
        
        # Detect communities using Louvain method
        communities = nx.community.louvain_communities(G)
        
        # Calculate network metrics
        centrality = nx.eigenvector_centrality_numpy(G)
        clustering = nx.clustering(G)
        
        return G, communities, centrality, clustering
    except:
        return None, [], {}, {}

def home(request):
    return render(request, 'home.html')

def view(request):
    return render(request, 'view.html')

def graph(request):
    try:
        df = pd.read_csv('data/network.csv')
        G = nx.Graph()
        
        # Build graph
        for _, row in df.iterrows():
            source = int(row['node'])
            targets = [int(t.strip()) for t in row['connections'].split(',')]
            G.add_edges_from([(source, target) for target in targets])
        
        # Calculate node metrics
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)
        
        # Create adjacency matrix for clustering
        adj_matrix = nx.to_numpy_array(G)
        n_clusters = 18  # Updated number of clusters
        
        # Perform spectral clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(adj_matrix)
        
        # Extended color palette for 18 groups with regional color coding
        colors = [
            # Frontal regions (reds)
            '#FF3366', '#FF4D4D', '#FF6B61',
            # Temporal regions (blues)
            '#3366FF', '#4D94FF', '#66B2FF',
            # Parietal regions (greens)
            '#33FF99', '#66FF66', '#99FF33',
            # Occipital regions (purples)
            '#9933FF', '#B266FF', '#CC99FF',
            # Motor and sensory regions (oranges)
            '#FF9933', '#FFB266', '#FFCC99',
            # Deep brain regions (cyans)
            '#33FFFF', '#66FFFF', '#99FFFF'
        ]
        
        # Create nodes with group information
        nodes = []
        max_importance = max((degrees[node] * 0.5 + betweenness[node] * 0.5) for node in G.nodes())
        
        for node in G.nodes():
            group_id = int(clusters[node-1])
            # Normalize importance to keep node sizes smaller
            importance = (degrees[node] * 0.5 + betweenness[node] * 0.5) / max_importance
            nodes.append({
                'id': node,
                'group': group_id,
                'color': colors[group_id],
                'size': 3 + (importance * 10),
                'degree': degrees[node]
            })
        
        # Sort links by group connectivity
        links = []
        for u, v in G.edges():
            links.append({
                'source': u,
                'target': v,
                'value': 1 if clusters[u-1] == clusters[v-1] else 0.5
            })
        
        return JsonResponse({
            'nodes': nodes,
            'links': links,
            'groups': n_clusters
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({'error': str(e)})

def draw_graph(request):
    graph_data = graph(request)
    return render(request, 'view.html', {'graph_data': graph_data})
