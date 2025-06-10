from django.urls import path, include

urlpatterns = [
    path("", include("map_publish_app.urls")),
]
