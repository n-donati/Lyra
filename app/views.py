from io import StringIO
import json
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from functools import lru_cache
from network_gen import generate_matrix
from .models import * 
from .utilities.extractCSV import extract_adjacency_matrix
from django.contrib import messages
from .utilities.extractCSV import extract_adjacency_matrix
from .openai import ChatGPT
import json

current_matrix = [[]]

def get_matrix(student_id):
    adj_graph = []
    return_matrix = []
    
    print("beginning")
    # matrix = Matrix.objects.get(user_id=student_id)
    nodes = Neuron.objects.filter(matrix_id=student_id)
    
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
    print("RETURN MATRIX", return_matrix)
                
    return return_matrix

# @csrf_exempt
# def generate_response(request):
#     if request.method == 'POST':
#         message = request.POST.get('message')
#         chatgpt = ChatGPT()  # No need to pass the API key here
#         response = chatgpt.get_response(message)  # Changed from generate_response to get_response
#         return JsonResponse({'response': response})


def demo_students():
    if not User.objects.exists():
        Matrix.objects.bulk_create([
            Matrix(name="Secundaria Sotelo"),
            Matrix(name="Carrera IFI"),
            Matrix(name="Preparatoria CBTIS"),
            Matrix(name="Universidad ITESM")
        ])
        User.objects.bulk_create([
            User(name="Juan", role="Student", matrix_id=Matrix.objects.get(name="Carrera IFI")),
            User(name="Maria", role="Student", matrix_id=Matrix.objects.get(name="Carrera IFI")),
            User(name="Pedro", role="Student", matrix_id=Matrix.objects.get(name="Preparatoria CBTIS")),
            User(name="Ana", role="Student", matrix_id=Matrix.objects.get(name="Universidad ITESM"))
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
        
    # print(data)
    
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
                    
# def process_matrix(matrix_data, student):
#     neuron_ids = []
    
#     matrix = Matrix(user_id=User.objects.get(id=student.id))
#     matrix.save()
    
    
#     for i, row in enumerate(matrix_data):
#         for j in range(len(row)):
#             value = float(row[j])
#             if i != j and value > 0:
#                 connection = Connection(
#                     neuron_id=Neuron.objects.get(id=neuron_ids[i]), 
#                     con_neuron_id=Neuron.objects.get(id=neuron_ids[j]),
#                 )
#                 connection.save()

def upload_to_database(adjacency_matrix, matrix_id):
    print(adjacency_matrix.shape)
    matrix = Matrix.objects.get(id=matrix_id)  # Retrieve the matrix instance by ID
    neuron_mapping = {}
    # Create neurons for each row in the adjacency matrix
    # neurons = []
    count = 0
    for name in adjacency_matrix.columns:
        neuron = Neuron(
            name=name,  # Use the name from the adjacency matrix
            color="Color",  # Example placeholder
            size=1.0,       # Example placeholder
            opacity=1.0,    # Example placeholder
            neuron_no= count,  # Optional, for indexing
            matrix_id=matrix
        )
        count += 1
        neuron.save()
        # neurons.append(neuron)
        neuron_mapping[name] = neuron  

    # Create connections based on the adjacency matrix
    for i, row_name in enumerate(adjacency_matrix.index):
        for j, col_name in enumerate(adjacency_matrix.columns):
            value = adjacency_matrix.iat[i, j]
            if row_name != col_name and int(value) > 0:  # Only create if there is a relation and not self
                connection = Connection(
                    neuron_id=neuron_mapping[row_name],     # Get neuron by name
                    con_neuron_id=neuron_mapping[col_name]  # Get connected neuron by name
                )
                connection.save()

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def view(request):
    global current_matrix
    demo_students()
    users = User.objects.all()
    matrices = Matrix.objects.all()
    
    if request.method == 'POST':
        matrix_id = None
        uploaded_file = None
        
        for key, value in request.FILES.items():
            if '.file' in key:
                matrix_id = key.split('.')[0]  
                uploaded_file = value
                
                
            if matrix_id and uploaded_file:
                # matrix = Matrix.objects.get(id=matrix_id)
                adjacency_matrix = extract_adjacency_matrix(uploaded_file)
                upload_to_database(adjacency_matrix, matrix_id)
                current_matrix = get_matrix(matrix_id)
                draw_graph(request)
            
            # if request.POST.get('student_id'):
            #     matrix_id = request.POST.get('student_id')
            #     current_matrix = get_matrix(matrix_id)
            #     draw_graph(request)
            #     return render(request, 'view.html', {"matrices": matrices, "users": users})
                
                # # first create matrix
                # adjacency_matrix = extract_adjacency_matrix(uploaded_file)
                # upload_to_database(adjacency_matrix, matrix_id)
                # global current_matrix 
                # print("reached")
                # current_matrix= get_matrix(matrix_id)
                # draw_graph(request)
                

                # messages.success(request, 'File uploaded successfully!')
                # return render(request, 'view.html', {"matrices": matrices, "users": users}) # Redirect to a new URL or refresh
    return render(request, 'view.html', {"matrices": matrices, "users": users})

def graph(request):
    global current_matrix
    try:
        # Generate a larger matrix with our desired parameters
        matrix = current_matrix
        print(matrix)
        
        # Convert matrix to networkx graph
        G = nx.Graph(np.array(matrix))
        
        # Calculate node metrics
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)
        
        # Create adjacency matrix for clustering
        adj_matrix = nx.to_numpy_array(G)
        n_clusters = min(18, len(matrix))  # Adjust clusters based on matrix size
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(adj_matrix)
        
        # Color palette remains the same
        colors = [
            '#FF00FF', '#FF33FF', '#FF66FF',  # Neon pinks
            '#00FFFF', '#33FFFF', '#66FFFF',  # Neon cyans
            '#00FF00', '#33FF33', '#66FF66',  # Neon greens
            '#FFFF00', '#FFFF33', '#FFFF66',  # Neon yellows
            '#FF6600', '#FF9933', '#FFCC66',  # Neon oranges
            '#6600FF', '#9933FF', '#CC66FF'   # Neon purples
        ]
        
        # Create nodes
        nodes = []
        max_importance = max((degrees[node] * 0.5 + betweenness[node] * 0.5) 
                           for node in G.nodes())
        
        for node in G.nodes():
            group_id = int(clusters[node])
            importance = (degrees[node] * 0.5 + betweenness[node] * 0.5) / max_importance
            nodes.append({
                'id': node + 1,  # Add 1 to match the JS expectation
                'group': group_id,
                'color': colors[group_id],
                'size': 3 + (importance ** 2 * 30),
                'degree': degrees[node],
                'label': node
            })
        
        # Create links
        links = []
        for u, v in G.edges():
            links.append({
                'source': u + 1,  # Add 1 to match the JS expectation
                'target': v + 1,  # Add 1 to match the JS expectation
                'value': 1 if clusters[u] == clusters[v] else 0.5
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
