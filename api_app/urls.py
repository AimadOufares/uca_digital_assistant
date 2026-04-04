from django.urls import path

from .views import ChatAPIView, TestView

urlpatterns = [
    path("test/", TestView.as_view(), name="api-test"),
    path("chat/", ChatAPIView.as_view(), name="api-chat"),
]
