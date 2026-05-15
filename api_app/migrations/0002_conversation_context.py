from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api_app", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversation",
            name="context_summary",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="conversation",
            name="context_meta",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
