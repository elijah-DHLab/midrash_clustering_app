from django.urls import path

from explorer import views

urlpatterns = [
    path("", views.index, name="index"),
    path("run/", views.run, name="run"),
    path("progress/<str:job_id>", views.progress, name="progress"),
    path("result/<str:job_id>", views.result, name="result"),
    path("download/<str:token>/<str:name>", views.download, name="download"),
]
