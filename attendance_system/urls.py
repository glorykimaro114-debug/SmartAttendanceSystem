from django.contrib import admin
from django.urls import path
from main import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard),
    path('home/', views.home),
    path('login/', views.login),
    path('dashboard/', views.dashboard),
    path('register/', views.register_student),
    path('capture/', views.capture, name='capture'),
    path('train/', views.train_model, name='train'),
    path('recognize/', views.recognize_face, name='recognize'),
    path('reports/', views.reports, name='reports'),
    path('export/excel/', views.export_excel, name='export_excel'),
    path('export/pdf/', views.export_pdf, name='export_pdf'),
]
