# Generated by Django 3.1 on 2020-08-26 03:45

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PredictionRequestModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sound', models.FileField(default=0, upload_to='server/bcdapp/reqsounds/')),
            ],
        ),
    ]