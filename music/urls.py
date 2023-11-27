from django.urls import path
from . import views

urlpatterns = [
  path('', views.index, name='index'),
  path('ajax', views.ajax, name='ajax'),
  path('detail/<int:page>', views.detail, name='detail'),
  path('edit/<int:page>', views.edit, name='edit'),
  path('emotion/<str:emotion>', views.emotion_folder, name='emotion_folder'),
  path('clear_database/', views.clear_database, name='clear_database'),
]