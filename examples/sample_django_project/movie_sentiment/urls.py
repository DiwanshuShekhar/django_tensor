from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^q/$', views.q, name='query'),
]