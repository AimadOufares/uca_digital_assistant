import json
import shutil
from pathlib import Path
from unittest.mock import patch

import portalocker
from django.contrib.auth import get_user_model
from django.conf import settings
from django.test import TestCase


class AdminDashboardAccessTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.staff_user = user_model.objects.create_user(
            username="staff",
            password="password123",
            is_staff=True,
        )

    def test_dashboard_metrics_requires_staff(self):
        response = self.client.get("/api/dashboard-metrics/")

        self.assertEqual(response.status_code, 403)

    def test_run_audit_requires_staff(self):
        response = self.client.post("/api/run-audit/data_audit/")

        self.assertEqual(response.status_code, 403)

    def test_dashboard_page_redirects_anonymous_users_to_admin_login(self):
        response = self.client.get("/admin-dashboard/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response["Location"])


class AdminDashboardReportTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.staff_user = user_model.objects.create_user(
            username="staff_reports",
            password="password123",
            is_staff=True,
        )
        self.client.force_login(self.staff_user)

    def test_dashboard_metrics_returns_latest_report_for_each_section(self):
        reports_dir = Path(settings.BASE_DIR) / "data_storage" / "test_reports_dashboard"
        reports_dir.mkdir(parents=True, exist_ok=True)
        try:
            older_data = reports_dir / "data_audit_older.json"
            newer_data = reports_dir / "data_audit_newer.json"
            rag_eval = reports_dir / "rag_eval_latest.json"

            older_data.write_text(json.dumps({"generated_at": "2026-04-10T10:00:00", "value": 1}), encoding="utf-8")
            newer_data.write_text(json.dumps({"generated_at": "2026-04-11T10:00:00", "value": 2}), encoding="utf-8")
            rag_eval.write_text(json.dumps({"generated_at": "2026-04-12T10:00:00", "score": 99}), encoding="utf-8")

            older_ts = 1_776_000_000
            newer_ts = 1_776_100_000
            rag_ts = 1_776_200_000
            older_data.touch()
            newer_data.touch()
            rag_eval.touch()
            older_data_stat = (older_ts, older_ts)
            newer_data_stat = (newer_ts, newer_ts)
            rag_eval_stat = (rag_ts, rag_ts)

            import os

            os.utime(older_data, older_data_stat)
            os.utime(newer_data, newer_data_stat)
            os.utime(rag_eval, rag_eval_stat)

            with patch("api_app.views._get_reports_dir", return_value=reports_dir):
                response = self.client.get("/api/dashboard-metrics/")
        finally:
            shutil.rmtree(reports_dir, ignore_errors=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data_audit"]["value"], 2)
        self.assertEqual(response.json()["data_audit"]["_report_file"], "data_audit_newer.json")
        self.assertEqual(response.json()["rag_eval"]["score"], 99)

    def test_run_audit_returns_conflict_when_same_task_is_already_locked(self):
        with patch(
            "api_app.views.portalocker.Lock",
            side_effect=portalocker.exceptions.AlreadyLocked(),
        ):
            response = self.client.post("/api/run-audit/data_audit/")

        self.assertEqual(response.status_code, 409)
        self.assertIn("déjà en cours", response.json()["error"])
