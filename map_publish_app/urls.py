from django.urls import path
from .views import route_crosswalk_view

urlpatterns = [
    path("route/crosswalk/", route_crosswalk_view),
]
