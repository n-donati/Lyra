from django.db import models
    
class User(models.Model):
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=100, default="USER")
    
    def __str__(self):
        return self.name
    
class Matrix(models.Model):
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def __str__(self):
        return "Matrix" + self.user_id

class Neuron(models.Model):
    name = models.CharField(max_length=100, default="")
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
        return "Neurons"
    
    