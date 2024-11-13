import csv
from io import StringIO
from django.shortcuts import render
from django.http import HttpResponse
from .models import * 

def read_csv(file):
    neuron_ids = []
    
    try:
        csv_file_content = file.read().decode('utf-8-sig')  
    except UnicodeDecodeError:
        csv_file_content = file.read().decode('utf-8')
    
    csv_data = StringIO(csv_file_content)
    reader = csv.reader(csv_data)
    
    matrix = Matrix(user_id=User.objects.create())
    matrix.save()
    
    for i, row in enumerate(reader):
        neuron = Neuron(
            size=0,
            color="Color",
            opacity=0.0,
            matrix_id=matrix, 
            neuron_no=i
        )
        neuron.save()
        neuron_ids.append(neuron.id)
        
    csv_data.seek(0)
    reader = csv.reader(csv_data)
    
    for i, row in enumerate(reader):
        for j in range(len(row)):
            value = float(row[j])
            if i != j and value > 0:
                connection = Connection(
                    neuron_id=Neuron.objects.get(id=neuron_ids[i]), 
                    con_neuron_id=Neuron.objects.get(id=neuron_ids[j]),
                )
                connection.save()

def home(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        
        if(csv_file):
            read_csv(csv_file)
        
    return render(request, 'home.html')
