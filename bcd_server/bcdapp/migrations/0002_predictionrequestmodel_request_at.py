# Generated by Django 3.1 on 2020-08-26 03:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bcdapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictionrequestmodel',
            name='request_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]