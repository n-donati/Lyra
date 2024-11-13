from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('view/', views.view, name='view'),
    path('graph/', views.graph, name='graph'),
    path('draw-graph/', views.draw_graph, name='draw_graph'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)