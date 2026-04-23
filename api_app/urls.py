from django.urls import path

from .views import (
    AdminDashboardAPIView,
    AdminDashboardPageView,
    ChatAPIView,
    ChatPageView,
    LiveHealthAPIView,
    ReadyHealthAPIView,
    TestView,
)

urlpatterns = [
    path("test/", TestView.as_view(), name="api-test"),
    path("chat/", ChatAPIView.as_view(), name="api-chat"),
    path("health/live/", LiveHealthAPIView.as_view(), name="api-health-live"),
    path("health/ready/", ReadyHealthAPIView.as_view(), name="api-health-ready"),
    path("ui/chat/", ChatPageView.as_view(), name="ui-chat"),
    path("admin-dashboard/", AdminDashboardPageView.as_view(), name="admin-dashboard"),
    path("dashboard-metrics/", AdminDashboardAPIView.as_view(), name="api-dashboard-metrics"),
]
