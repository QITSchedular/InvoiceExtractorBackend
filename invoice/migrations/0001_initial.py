# Generated by Django 5.1.3 on 2024-11-08 11:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Template',
            fields=[
                ('transId', models.AutoField(primary_key=True, serialize=False)),
                ('value', models.CharField(max_length=255)),
                ('template', models.JSONField()),
            ],
        ),
    ]
