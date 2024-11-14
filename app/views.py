from io import StringIO
import json
from django.shortcuts import render
from django.http import JsonResponse
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from functools import lru_cache
from network_gen import generate_matrix
from .models import * 
from .utilities.extractCSV import extract_adjacency_matrix
from django.contrib import messages

current_matrix = [[]]

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

def home(request):
    return render(request, 'home.html')

def view(request):
    # global current_matrix
    # demo_students()
    # students = User.objects.filter(role="STUDENT")
    
    # if request.method == 'POST':
    #     student_id = None
    #     uploaded_file = None
        
    #     for key, value in request.FILES.items():
    #         if '.file' in key:
    #             student_id = key.split('.')[0]  
    #             uploaded_file = value
                      
    #     if student_id and uploaded_file:
    #         student = User.objects.get(id=student_id)
    #         matrix = extract_adjacency_matrix(uploaded_file)
    #         read_csv(matrix, student)
    #         # matrix_data = json.load(uploaded_file)
    #         # process_matrix(matrix_data, student)
        
    #     if request.POST.get('student_id'):
    #         student_id = request.POST.get('student_id')
    #         current_matrix = get_matrix(student_id)
    #         draw_graph(request)
    #         return render(request, 'view.html', {"students": students, "student_id": student_id})
    try:
        demo_students()
    except:
        pass
    matrices = Matrix.objects.all()
    users = User.objects.all()
    
    if request.method == 'POST':
        # Get matrix_id directly from POST data
        matrix_id = request.POST.get('matrix_id')
        uploaded_file = request.FILES.get('file')  # Get the uploaded file
        if matrix_id and uploaded_file:
            try:
                matrix = Matrix.objects.get(id=matrix_id)
                read_csv(uploaded_file, matrix)  # Your function to handle the CSV

                messages.success(request, 'File uploaded successfully!')
                return render(request, 'view.html', {}) # Redirect to a new URL or refresh
            except Matrix.DoesNotExist:
                messages.error(request, 'Matrix not found.')
            except Exception as e:
                messages.error(request, f'Error uploading file: {str(e)}')
    
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
                'color': colors[group_id % len(colors)],
                'size': 3 + (importance * 10),
                'degree': degrees[node]
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
