from django.shortcuts import render
from django_pandas.io import read_frame
from predictor.models import Stock, TrainingSession
from django.utils import timezone
from django.http import JsonResponse
from django.db.models import Q
from datetime import timedelta
from utils.utils import get_forecast_plots, get_evaluation_plot

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
    start_date = now - timedelta(days=90)
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
        ctx_data['training_session_objs'] = TrainingSession.objects.all()
        return render(request, 'predictor/forecast.html', ctx_data)


def evaluations(request):
    """
    The evaluate page

    :param request:
    :return:
    """
    # Which session are we predicting with?
    training_session_id = request.POST.get('training_session_id', 'ba319bc2-9345-4a6d-8226-f42641a2fac8')

    graph_0, layout_0 = get_evaluation_plot(training_session_id, eval_type='train')
    graph_1, layout_1 = get_evaluation_plot(training_session_id, eval_type='test')

    # append all charts to the figures list
    figures = [dict(data=graph_0, layout=layout_0),
               dict(data=graph_1, layout=layout_1)]

    # plot ids for the html id tag
    ids = [f'figure-{i}' for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figures_json = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    ctx_data = {'ids': ids,
                'figuresJSON': figures_json}

    if request.is_ajax():
        return JsonResponse(ctx_data)
    else:
        ctx_data['training_session_objs'] = TrainingSession.objects.all()
        return render(request, 'predictor/evaluations.html', ctx_data)
