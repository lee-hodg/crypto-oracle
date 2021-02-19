from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import TrainingSession, Stock, StockPrediction


class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'name',
        'start_date',
        'end_date',
        'epochs',
        'window_length',
        'output_size',
        'neurons',
        'scaler'
    ]

    class Meta:
        model = TrainingSession


admin.site.register(TrainingSession, TrainingSessionAdmin)


class StockAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'dt',
        'name',
        'open',
        'high',
        'low',
        'close',
        'volume'
    ]

    class Meta:
        model = Stock


admin.site.register(Stock, StockAdmin)


class StockPredictionAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'training_session',
        'dt',
        'prediction'
    ]

    class Meta:
        model = StockPrediction


admin.site.register(StockPrediction, StockPredictionAdmin)
