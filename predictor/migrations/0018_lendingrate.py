# Generated by Django 3.1.6 on 2021-02-25 15:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0017_auto_20210223_1505'),
    ]

    operations = [
        migrations.CreateModel(
            name='LendingRate',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('platform', models.CharField(choices=[('ftx', 'FTX'), ('kucoin', 'KuCoin')], default='ftx', max_length=20)),
                ('coin', models.CharField(max_length=20)),
                ('dt', models.DateTimeField(db_index=True)),
                ('previous', models.DecimalField(decimal_places=10, max_digits=19, null=True)),
                ('estimate', models.DecimalField(decimal_places=10, max_digits=19, null=True)),
            ],
            options={
                'unique_together': {('platform', 'coin', 'dt')},
            },
        ),
    ]