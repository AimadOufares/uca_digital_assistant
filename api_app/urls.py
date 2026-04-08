from django.urls import path

from .views import ChatAPIView, TestView, ChatPageView, AdminDashboardPageView, AdminDashboardAPIView

urlpatterns = [
    path("test/", TestView.as_view(), name="api-test"),
    path("chat/", ChatAPIView.as_view(), name="api-chat"),
    path("ui/chat/", ChatPageView.as_view(), name="ui-chat"),
    path("admin-dashboard/", AdminDashboardPageView.as_view(), name="admin-dashboard"),
    path("dashboard-metrics/", AdminDashboardAPIView.as_view(), name="api-dashboard-metrics"),
]
