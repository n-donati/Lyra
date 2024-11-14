from io import StringIO
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import networkx as nx
import pandas as pd
from functools import lru_cache
import numpy as np
from sklearn.cluster import KMeans
from .models import * 
from .utilities.extractCSV import extract_adjacency_matrix

def get_matrix(student_id):
    adj_graph = []
    return_matrix = []
    
    matrix = Matrix.objects.get(user_id=student_id)
    nodes = Neuron.objects.filter(matrix_id=matrix)
    
    for i in range(len(nodes)):
        adj_graph.append([])
        connections = Connection.objects.filter(neuron_id=nodes[i])
        for j in range(len(connections)):
            adj_graph[i].append(connections[j].con_neuron_id.id)
    
    # print(len(nodes))
    # print(len(adj_graph))
    
    for i in range(len(nodes)):
        k = 0
        return_matrix.append([])
        for j in range(len(nodes)):
            if i == j:
                return_matrix[i].append(0)
            elif k >= len(adj_graph[i]) - 1:
                return_matrix[i].append(0)
            elif Neuron.objects.get(id=adj_graph[i][k]).neuron_no == j:
                return_matrix[i].append(1)
                k+=1
            else:
                return_matrix[i].append(0)
                
    return return_matrix
        

def demo_students():
    if not User.objects.exists():
        User.objects.bulk_create([
            User(name="Edgar Martínez", role="STUDENT"),
            User(name="André Robles", role="STUDENT"),
            User(name="Nicolás Donati", role="STUDENT"),
            User(name="Emily Costain", role="STUDENT"),
            User(name="Nicolas de Alba", role="STUDENT"),
        ])

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

def read_csv(matrix, student):
    neuron_ids = []
    
    m = Matrix(user_id=User.objects.get(id=student.id))
    m.save()
    
    if isinstance(matrix, Matrix):
        data = matrix.get_data()  
    else:
        data = matrix
        
    print(data)
        
    for i in range(len(matrix)):
        neuron = Neuron(
            size=0,
            color="Color",
            opacity=0.0,
            matrix_id=m, 
            neuron_no=i,
            name=matrix[i][len(matrix)]
        )
        neuron.save()
        neuron_ids.append(neuron.id)
        
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if j < len(matrix[i]) - 1:
                value = float(matrix[i][j])
                if i != j and value > 0:
                    connection = Connection(
                        neuron_id=Neuron.objects.get(id=neuron_ids[i]), 
                        con_neuron_id=Neuron.objects.get(id=neuron_ids[j]),
                    )
                    connection.save()

def home(request):
    return render(request, 'home.html')

def view(request):
    demo_students()
    students = User.objects.filter(role="STUDENT")
    
    if request.method == 'POST':
        student_id = None
        uploaded_file = None
        
        for key, value in request.FILES.items():
            if '.file' in key:
                student_id = key.split('.')[0]  
                uploaded_file = value
                      
        if student_id and uploaded_file:
            student = User.objects.get(id=student_id)
            matrix = extract_adjacency_matrix(uploaded_file)
            read_csv(matrix, student)
        
        if request.POST.get('student_id'):
            student_id = request.POST.get('student_id')
            get_matrix(student_id)
            return render(request, 'view.html', {"students": students, "student_id": student_id})
    
    return render(request, 'view.html', {"students": students})

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

def settings_view(request):
    if request.method == 'POST':
        selected_option = request.POST.get('selector')
        print(f"Selected option: {selected_option}")
        # Handle the selected option
        # ...existing code...
        print("ahuevo verga")
    return render(request, 'view.html')
