from django.urls import path
from .views import predict, hello

urlpatterns = [
    path('predict/', predict),
    path('hello/', hello),
]

