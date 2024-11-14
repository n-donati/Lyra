from django.db import models

class Matrix(models.Model):
    name = models.CharField(max_length=100, default="myMATRIX", unique=True)
    
    def __str__(self):
        return "Matrix" + self.name

class User(models.Model):
    name = models.CharField(max_length=100, default="USER")
    role = models.CharField(max_length=100, default="USER")
    matrix_id = models.ForeignKey(Matrix, on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name

class Neuron(models.Model):
    name = models.CharField(max_length=100, default="Neuron")
    color = models.CharField(max_length=100)
    size = models.FloatField()
    opacity = models.FloatField()
    neuron_no = models.IntegerField()
    matrix_id = models.ForeignKey(Matrix, on_delete=models.CASCADE)
    
    def __str__(self):
        return "Neuron" + self
    
class Connection(models.Model):
    neuron_id = models.ForeignKey(Neuron, related_name="neuron_connections", on_delete=models.CASCADE)
    con_neuron_id = models.ForeignKey(Neuron, related_name="con_neuron_connections", on_delete=models.CASCADE)
    
    def __str__(self):
        return self.neuron_id + self.con_neuron_id
    
    