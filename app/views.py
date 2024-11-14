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
from .models import * 
from .utilities.extractCSV import extract_adjacency_matrix
from django.contrib import messages
from .openai import ChatGPT
import json

current_matrix = [[]]

def get_matrix(student_id):
    # Get all nodes in one query and create a lookup dictionary
    nodes = list(Neuron.objects.filter(matrix_id=student_id))
    node_lookup = {node.id: node.neuron_no for node in nodes}
    n = len(nodes)
    
    # Initialize empty matrix
    return_matrix = [[0] * n for _ in range(n)]
    
    # Get all connections in a single query
    connections = Connection.objects.filter(
        neuron_id__matrix_id=student_id
    ).select_related('neuron_id', 'con_neuron_id')
    
    # Fill matrix using the connections
    for conn in connections:
        from_idx = node_lookup[conn.neuron_id.id]
        to_idx = node_lookup[conn.con_neuron_id.id]
        return_matrix[from_idx][to_idx] = 1
        
    return return_matrix

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

def upload_to_database(adjacency_matrix, matrix_id, opacities):
    
    print(adjacency_matrix.shape)
    matrix = Matrix.objects.get(id=matrix_id)
    neuron_mapping = {}
    count = 0
    
    # Create all neurons first
    for name in adjacency_matrix.columns:
        neuron = Neuron(
            name=name,  # Use the name from the adjacency matrix
            color="Color",  # Example placeholder
            size=1.0,       # Example placeholder
            opacity=opacities.iloc[count + 4],    # Example placeholder
            neuron_no= count,  # Optional, for indexing
            matrix_id=matrix
        )
        neuron.save()
        neuron_mapping[name] = neuron
        count += 1

    # Create connections only where value > 0
    connections_to_create = []
    for row_name, row in adjacency_matrix.iterrows():
        # Use boolean indexing to find non-zero connections
        connections = row[row > 0]
        for col_name in connections.index:
            if row_name != col_name:  # Skip self-connections
                connections_to_create.append(Connection(
                    neuron_id=neuron_mapping[row_name],
                    con_neuron_id=neuron_mapping[col_name]
                ))
    
    # Bulk create all connections at once
    if connections_to_create:
        Connection.objects.bulk_create(connections_to_create)

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def view(request):
    global current_matrix
    demo_students()
    users = User.objects.all()
    matrices = Matrix.objects.all()
    context = {"matrices": matrices, "users": users}
    matrix = ""
    
    if request.method == 'POST':
        if request.POST.get('message'):
            message = request.POST.get('message')
            chatgpt = ChatGPT()  
            gpt_response = chatgpt.get_response(message)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'response': gpt_response})
            context['gpt_response'] = gpt_response
            
        matrix_id = None
        uploaded_file = None
        
        for key, value in request.FILES.items():
            if '.file' in key:
                matrix_id = key.split('.')[0]  
                uploaded_file = value
                
                
            if matrix_id and uploaded_file:
                # matrix = Matrix.objects.get(id=matrix_id)
                adjacency_matrix, opacities = extract_adjacency_matrix(uploaded_file)
                upload_to_database(adjacency_matrix, matrix_id, opacities)
                current_matrix = get_matrix(matrix_id)
                draw_graph(request)
                matrix = Matrix.objects.get(id=matrix_id)
                context = {"matrices": matrices, "users": users, "matrix": matrix}

                #     return JsonResponse({'matrix': matrix})
                # context['matrix'] = matrix
                
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
    return render(request, 'view.html', context)

def graph(request):
    global current_matrix
    try:
        # Cargar y preparar la matriz de adyacencia actual
        matrix = current_matrix
        G = nx.Graph(np.array(matrix))  # Crear el grafo desde la matriz de adyacencia
        
        # Generar un layout de red para distribuir los nodos
        pos = nx.spring_layout(G, seed=42)  # Usar spring_layout para una disposición compacta
        
        # Calcular métricas para los nodos
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)
        
        # Preparar la matriz para clustering
        adj_matrix = nx.to_numpy_array(G)
        n_clusters = min(8, len(matrix))  # Ajustar el número de clusters
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(adj_matrix)
        
        # Definir una paleta de colores
        colors = [
            '#3b82f6', '#10b981', '#f59e0b',  # Neon pinks
            '#ef4444', '#ef4444', '#f97316',  # Neon cyans
            '#6366f1', '#8b5cf6'
        ]
        
        # Crear nodos
        nodes = []
        max_importance = max((degrees[node] * 0.5 + betweenness[node] * 0.5) 
                             for node in G.nodes())
        
        i = 1
        opacities = Neuron.objects.values_list('opacity', flat=True)
        range = max(opacities) - min(opacities)
        for node in G.nodes():
            opacity = Neuron.objects.get(id=i).opacity
            opacity = opacity * range / 1000
            print(type(Neuron.objects.get(id=i).opacity))
            print(Neuron.objects.get(id=i).opacity)
            print("opacity", opacity, "\n\n")
            if opacity < .25:
                opacity = .25
                
            group_id = int(clusters[node])
            importance = (degrees[node] * 0.5 + betweenness[node] * 0.5) / max_importance
            nodes.append({
                'id': node + 1,
                'group': group_id,
                'color': colors[group_id % len(colors)],
                'size': 10 + (importance ** 2.5 * 20),
                'degree': degrees[node],
                'label': Neuron.objects.get(id=i).name,
                'font_size': "2px",
                'x': pos[node][0] * 1000,  # Usar posiciones del layout
                'y': pos[node][1] * 1000,
                'opacity': opacity
            })
            i += 1
        
        # Crear enlaces
        links = []
        for u, v in G.edges():
            links.append({
                'source': u + 1,
                'target': v + 1,
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
        print("ahuevo verga")
    return render(request, 'view.html')
