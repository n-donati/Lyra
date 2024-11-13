import csv
from io import StringIO
from django.shortcuts import render
from django.http import HttpResponse
from .models import * 
def home(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        
        try:
            csv_file_content = csv_file.read().decode('utf-8-sig')  
        except UnicodeDecodeError:
            csv_file_content = csv_file.read().decode('utf-8')
        
        csv_data = StringIO(csv_file_content)
        reader = csv.reader(csv_data)
        
        for i, row in enumerate(reader):
            for j in range(len(row)):
                value = float(row[j])
                print(f"Coordenadas (i={i}, j={j}): {value}")
        
        return render(request, 'home.html', {'message': 'Archivo procesado correctamente'})
    return render(request, 'home.html')
