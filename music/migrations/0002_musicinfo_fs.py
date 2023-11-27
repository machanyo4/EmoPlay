# Generated by Django 4.1 on 2022-08-29 05:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('music', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='musicinfo',
            name='fs',
            field=models.JSONField(default=[{'Happy': 0, 'Melancholy': 0, 'Relaxed': 0, 'Tense': 100}]),
            preserve_default=False,
        ),
    ]
