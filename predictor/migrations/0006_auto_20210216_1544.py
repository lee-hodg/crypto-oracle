# Generated by Django 3.1.6 on 2021-02-16 15:44

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0005_auto_20210216_1543'),
    ]

    operations = [
        migrations.RenameField(
            model_name='trainingsession',
            old_name='uuid',
            new_name='id',
        ),
    ]
