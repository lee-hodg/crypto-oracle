from django.shortcuts import render
from django_pandas.io import read_frame
from predictor.models import Stock, TrainingSession, LendingRate
from django.utils import timezone
from django.http import JsonResponse
from django.db.models import Q
from datetime import timedelta
from utils.utils import get_forecast_plots, get_evaluation_plot, get_lending_rates

import plotly.graph_objs as go
import json
import plotly

import logging

logger = logging.getLogger(__name__)


def price_graph(df):
    """
    Line chart showing the price over some datetime

    Args:
        :df: the dataframe of prices and volumes over time
    Returns:
        :return: the plot graph obj and layout
    """
    graph = []
    for col in ['close']:
        graph.append(go.Scatter(x=df.dt.tolist(),
                                y=df[col].tolist(),
                                mode='lines',
                                name=col)
                    )

    layout = dict(title=f'Evolution of price over time',
                  xaxis=dict(title='Date',
                             autotick=True),
                  yaxis=dict(title="Price"),
                  )

    return graph, layout


def vol_graph(df):
    """
    Line chart showing the vol over some datetime

    Args:
        :df: the dataframe of prices/vol over time
    Returns:
        :return: the plot graph obj and layout
    """
    graph = []
    for col in ['volume']:
        graph.append(go.Bar(x=df.dt.tolist(),
                            y=df[col].tolist(),
                            name=col,
                            width=1000 * 3600 * 24 * 1,
                            ))

    layout = dict(title=f'Evolution of volume over time',
                  xaxis=dict(title='Date',
                             autotick=True),
                  yaxis=dict(title="Volume"),
                  )

    return graph, layout


def index(request):
    """
    The index page

    :param request:
    :return:
    """
    now = timezone.now()
    start_date = now - timedelta(days=5)
    end_date = now

    qs = Stock.objects.filter(Q(dt__gte=start_date) & Q(dt__lte=end_date)).order_by('dt')
    df = read_frame(qs)

    graph_0, layout_0 = price_graph(df)
    graph_1, layout_1 = vol_graph(df)

    # append all charts to the figures list
    figures = [dict(data=graph_0, layout=layout_0),
               dict(data=graph_1, layout=layout_1)]

    # plot ids for the html id tag
    ids = [f'figure-{i}' for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figures_json = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    ctx_data = {'ids': ids,
                'figuresJSON': figures_json}

    print(ids)
    return render(request, 'predictor/index.html', ctx_data)


def forecast(request):
    """
    The forecast page

    :param request:
    :return:
    """
    # Which session are we predicting with?
    training_session_id = request.POST.get('training_session_id', 'ba319bc2-9345-4a6d-8226-f42641a2fac8')

    # The date range
    start_date = request.POST.get('start_date', timezone.now() - timedelta(days=30))
    end_date = request.POST.get('end_date', timezone.now())

    logger.debug(f'Plot training session id {training_session_id} from {start_date} to {end_date}')
    graph_0, layout_0 = get_forecast_plots(training_session_id, start_date, end_date)

    # append all charts to the figures list
    figures = [dict(data=graph_0, layout=layout_0)]

    # plot ids for the html id tag
    ids = [f'figure-{i}' for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figures_json = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    ctx_data = {'ids': ids,
                'figuresJSON': figures_json}

    if request.is_ajax():
        return JsonResponse(ctx_data)
    else:
        ctx_data['training_session_objs'] = TrainingSession.objects.filter(evaluation_session=False)
        return render(request, 'predictor/forecast.html', ctx_data)


def evaluations(request):
    """
    The evaluate page

    :param request:
    :return:
    """
    # Which session are we predicting with?
    training_session_id = request.POST.get('training_session_id', 'ba319bc2-9345-4a6d-8226-f42641a2fac8')

    (graph_a_0, layout_a_0), (graph_a_1, layout_a_1) = get_evaluation_plot(training_session_id, eval_type='train')
    (graph_b_0, layout_b_0), (graph_b_1, layout_b_1) = get_evaluation_plot(training_session_id, eval_type='test')

    # append all charts to the figures list
    figures = [dict(data=graph_a_0, layout=layout_a_0),
               dict(data=graph_a_1, layout=layout_a_1),
               dict(data=graph_b_0, layout=layout_b_0),
               dict(data=graph_b_1, layout=layout_b_1)
               ]

    # plot ids for the html id tag
    ids = [f'figure-{i}' for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figures_json = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    ctx_data = {'ids': ids,
                'figuresJSON': figures_json}

    if request.is_ajax():
        return JsonResponse(ctx_data)
    else:
        ctx_data['training_session_objs'] = TrainingSession.objects.filter(evaluation_session=True)
        return render(request, 'predictor/evaluations.html', ctx_data)


def lending_rate(request):
    """
    The lending rates

    :param request:
    :return:
    """
    # Which platform are we plotting? eg ftx
    platform = request.POST.get('platform', 'ftx')
    period = request.POST.get('period', 'Annual')
    coin = request.POST.getlist('coin[]', None)

    # The date range
    start_date = request.POST.get('start_date', timezone.now() - timedelta(days=30))
    end_date = request.POST.get('end_date', timezone.now())

    logger.debug(f'Plot platform {platform}, coin {coin}, period {period} from {start_date} to {end_date}')
    graph_0, layout_0 = get_lending_rates(platform, coin, period, start_date, end_date)

    # append all charts to the figures list
    figures = [dict(data=graph_0, layout=layout_0)]

    # plot ids for the html id tag
    ids = [f'figure-{i}' for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figures_json = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    ctx_data = {'ids': ids,
                'figuresJSON': figures_json}

    if request.is_ajax():
        return JsonResponse(ctx_data)
    else:
        # These are coins on latest dt sorted by biggest est first
        try:
            latest_lr = LendingRate.objects.latest('dt')
            ctx_data['coins'] = list(LendingRate.objects.filter(
                dt=latest_lr.dt).order_by('-estimate').values_list('coin', flat=True).distinct())
        except LendingRate.DoesNotExist:
            ctx_data['coins'] = []
        ctx_data['platforms'] = list(LendingRate.PLATFORMS._display_map.keys())
        ctx_data["selected_period"] = period
        ctx_data["selected_platform"] = platform
        ctx_data['periods'] = {'Hourly', 'Daily', 'Annual'}
        return render(request, 'predictor/lending_rates.html', ctx_data)