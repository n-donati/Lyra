import csv
from io import StringIO
from django.shortcuts import render
from django.http import HttpResponse
from .models import * 

def read_csv(file):
    try:
        csv_file_content = file.read().decode('utf-8-sig')  
    except UnicodeDecodeError:
        csv_file_content = file.read().decode('utf-8')
    
    csv_data = StringIO(csv_file_content)
    reader = csv.reader(csv_data)
    
    for i, row in enumerate(reader):
        for j in range(len(row)):
            value = float(row[j])
            print(f"Coordenadas (i={i}, j={j}): {value}")

def home(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        
        if(csv_file):
            read_csv(csv_file)
        
    return render(request, 'home.html')
