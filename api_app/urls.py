from django.urls import path

from .views import (
    AdminDashboardAPIView,
    AdminDashboardPageView,
    ChatAPIView,
    ChatPageView,
    DriveDocumentDetailAPIView,
    DriveDocumentsAPIView,
    DriveEvaluateAPIView,
    DriveRebuildAPIView,
    LatestAdminReportAPIView,
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
    path("drive-documents/", DriveDocumentsAPIView.as_view(), name="api-drive-documents"),
    path("drive-documents/<path:filename>/", DriveDocumentDetailAPIView.as_view(), name="api-drive-document-detail"),
    path("drive-rebuild/", DriveRebuildAPIView.as_view(), name="api-drive-rebuild"),
    path("drive-evaluate/", DriveEvaluateAPIView.as_view(), name="api-drive-evaluate"),
    path("reports/<str:kind>/latest/", LatestAdminReportAPIView.as_view(), name="api-report-latest"),
]
