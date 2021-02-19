from django.urls import path
from django.shortcuts import redirect

from . import views


urlpatterns = [
    path('', lambda request: redirect('index/', permanent=True)),
    path("index/", views.index, name="predictor_index"),
    path("forecast/", views.forecast, name="predictor_forecast"),
    path("evaluations/", views.evaluations, name="predictor_evaluations"),

]
