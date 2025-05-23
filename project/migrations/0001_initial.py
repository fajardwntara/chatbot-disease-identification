# Generated by Django 3.2.4 on 2022-05-18 15:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Answer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('answer', models.CharField(max_length=2000)),
            ],
            options={
                'verbose_name': 'Answer',
                'verbose_name_plural': 'Answer',
            },
        ),
        migrations.CreateModel(
            name='Context',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('context', models.CharField(max_length=2000)),
            ],
            options={
                'verbose_name': 'Context',
                'verbose_name_plural': 'Context',
            },
        ),
        migrations.CreateModel(
            name='Question',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.CharField(max_length=2000)),
            ],
            options={
                'verbose_name': 'Question',
                'verbose_name_plural': 'Question',
            },
        ),
    ]
