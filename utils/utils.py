"""
Collection of helper functions
"""
import tensorflow as tf
import logging
import argparse
import numpy as np
import psycopg2
import pandas as pd
import sys
import pickle
import plotly.graph_objs as go
from django_pandas.io import read_frame

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import stats
from dateutil.tz import tzutc
from dateutil.parser import parse
from django.utils.timezone import make_aware
from django.conf import settings
from predictor.models import TrainingSession, LendingRate
logger = logging.getLogger(__name__)


# The database parameters
DB_PARAMS = {'host': settings.DATABASES['default']['HOST'],
             'port': settings.DATABASES['default']['PORT'],
             'database': settings.DATABASES['default']['NAME'],
             'user': settings.DATABASES['default']['USER'],
             'password': settings.DATABASES['default']['PASSWORD']
             }


def valid_date(s):
    try:
        return make_aware(parse(s, tzinfos={'UTC': tzutc}))
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def connect(db_params):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        logger.debug('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**db_params)
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(error)
        sys.exit(1)
    logger.debug("Connection successful")
    return conn


def resample(open_prices, high_prices, low_prices, close_prices, volumes, period='H'):
    """Converts daily OHLCV prices to OHLCV prices in the period

    For example we could turn minute-wise data into hourly data or daily.

    Parameters
    ----------
    open_prices : DataFrame
        Minute open prices
    high_prices : DataFrame
       Minute high prices
    low_prices : DataFrame
        Minute low prices
    close_prices : DataFrame
        Minute close prices
    volumes:
        Minute volumes

    period: the resample period e.g W, H, D etc

    Returns
    -------
        The tuple of resamples series
    """

    # TODO: Implement Function

    open_prices = open_prices.resample(period).first()
    high_prices = high_prices.resample(period).max()
    low_prices = low_prices.resample(period).min()
    close_prices = close_prices.resample(period).last()
    volumes = volumes.resample(period).mean()

    return open_prices, high_prices, low_prices, close_prices, volumes


def split_data(data, training_size=0.8):
    """
    Split the data into training and test sets
    We want to preserve order and not shuffle at this stage as past points will be used to predict next in the sequence
    The test set will represent the unseen future
    However the windowed dataset will shuffle around each windows and along with target label (see that func)

    Params:
        data: the dataset
        training_sie: the split e.g. 0.8 means 80% of data is used in the training set
    """
    return data[:int(training_size*len(data))], data[int(training_size*len(data)):]


def windowed_dataset(series, shuffle_buffer, window_len, batch_size, window_scaling=False):
    """
    If we have a series like [1,2,3,4,5,6]
    We want to split it into windows, e.g. we take previous value of the series
    as the X input features and want to predict the following value as output Y
    e.g. [1,2] - > 3, so what we want to do is split the data into windows
    of length window_len + 1 (the +1 acconts for the label)
    E.g.
      [1, 2, 3]
      [2, 3, 4]
      [3, 4, 5]
      [4, 5, 6]


    Using tensorflow to do this for large datasets as it is more memory efficient.

    We shuffle the data to avoid bias

    Finally we split the example into input/target

    [[1, 2], [3]]
    .
    .

    so it's appropriate to feed into the model.fit

    and we batch it into batches of batch_size

    Params:
        series: the series upon which we perform the windowing
        shuffle_buffer: size of the buffer when shuffling
            (see https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#shuffle)
        window_len; how many previous elements to take into account when predicting the next
        batch_size: https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#batch
    """
    # Initially the data is (1188,) expand dims to TensorShape([1188, 1])
    series = tf.expand_dims(series, axis=-1)

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    # will be an iterable of tf.Tensor([998.325], shape=(1,), dtype=float32),...
    ds = tf.data.Dataset.from_tensor_slices(series)

    # https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
    # The +1 accounts for the label too. Create a bunch of windows over our series
    # If we started with ds = tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
    # then ds = ds.window(3, shift=1, drop_remainder=False) would lead
    # to [1,2,3], [2, 3, 4], [3, 4, 5], [4, 5], [5] whereas
    # drop_remainder=True) => [1,2,3], [2, 3, 4], [3, 4, 5]
    # Remember the first window_len are our training data and the 1 is
    # the target/label
    # Could also do this with pandas shift
    ds = ds.window(window_len + 1, shift=1, drop_remainder=True)
    # for w in ds:
    #    print(list(w.as_numpy_iterator()))

    # Maps map_func across this dataset and flattens the result
    ds = ds.flat_map(lambda w: w.batch(window_len + 1))

    # Instead of standard scaling all the data, sometimes people
    # normalize the window itself wrt to initial element
    # by default we do not do this as I found it gives bad results
    def normalize_window(w):
        return (w/w[0]) -1
    if window_scaling:
        ds = ds.map(normalize_window)

    # randomize order to remove biases
    ds = ds.shuffle(shuffle_buffer)

    # Collect the inputs and the target label
    ds = ds.map(lambda w: (w[:-1], w[-1]))

    return ds.batch(batch_size).prefetch(1)


def preprocessing(df, training_session, start_date=None, end_date=None, column_name='close', forecast=False):
    """
        * Data preprocessing.
        * First keep only data between the start and end date
        * Get the series (e.g. close prices)
        * Split into training and test sets
        * Do the type of scaling we desire, e.g. with StandardScaler or MinMaxScaler.
        * We fit to the training set (and not separately to the test as that would be a bias too).
        * We then have to transform the series to the 2D array the scaler expects  and back again to 1D
        * Finally we window the dataset to provide the feature/target data to train on

    :param df: The dataframe with prices/vol data
    :param training_session: the training session
    :param start_date: Period to start training from
    :param end_date: Period to end training
    :param column_name:
    :param forecast: is this for a forecast (no test/train split)

    :return:
         scaler - the scalar (use it to do inverse transform later)

         windowed_training_data - the windowed dataset and target labels to train the NN on

         training_data - the scaled (but not windowed) training data
         test_data - the scaled (but not windowed) test data

         training_dates - the date series for training set (useful for plots later)
         test_dates - the date series for test set (useful
    """

    # Date range of interest
    temp_df = df
    if start_date is not None:
        temp_df = temp_df[temp_df.index >= start_date]
    if end_date is not None:
        temp_df = temp_df[temp_df.index <= end_date]

    if forecast:
        # Simply return the data
        return temp_df

    # The relevant price data, e.g. close prices
    prices_df = temp_df[column_name]

    # Split into training/test datasets
    training_df, test_df = split_data(prices_df, training_size=training_session.training_size)

    # Want to normalize the log returns (must use same scaler on test and train
    # since not supposed to know about the test set)
    scaler = training_session.scaler
    logger.debug(f'Using scaler {scaler}')
    if scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'robust':
        sc = RobustScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    else:
        sc = None

    window_scaling = training_session.scaler == 'window'

    if sc is not None:
        # Fit on training, transform only the test
        training_data = sc.fit_transform(training_df.values.reshape(-1, 1)).flatten()
        if len(test_df):
            # sometimes no test data for prod sessions
            test_data = sc.transform(test_df.values.reshape(-1, 1)).flatten()
        else:
            test_data = test_df.values
        # Remember sc.inverse_transform should transform back the data too so we return the scaler too
    else:
        training_data = training_df.values
        test_data = test_df.values

    # Windowed/batched training data to feed the model
    windowed_training_data = windowed_dataset(training_data, training_session.shuffle_buffer_size,
                                              training_session.window_length, training_session.batch_size,
                                              window_scaling=window_scaling)

    # This will help with the displaying of results etc
    # Training and test dates for plotting comparisons
    # Note the first prediction will be at index window_len of the dataset
    training_dates = training_df.iloc[training_session.window_length:].index
    test_dates = test_df.iloc[training_session.window_length:].index

    return sc, windowed_training_data, training_data, test_data, training_dates, test_dates


def load_data(training_session, start_date=None, end_date=None, forecast=False):
    """

    * Read the data from the database into dataframe
    * Clean it up by dropping columns, sorting, indexing, ensuring correct dtypes
    * Do more pre-processing involving scaling with some scaler, windowing and batching

    :param training_session: the django model instance representing this training session, includes meta params
                    such as number of epochs, intervals, neurons of layers etc
    :param start_date:
    :param end_date:
    :param forecast: if forecast then prepare the data without training/test split and with window


    :return: The result of preprocessing:
        scaler - the fitted scaler (use it to do inverse transform later)

         windowed_training_data - the windowed dataset and target labels to train the NN on

         training_data - the scaled (but not windowed) training data
         test_data - the scaled (but not windowed) test data

         training_dates - the date series for training set (useful for plots later)
         test_dates - the date series for test set (useful for plots later)
    """
    conn = connect(DB_PARAMS)
    df = pd.read_sql_query('select * from "predictor_stock"', con=conn)

    # Cleanup
    df.drop(columns=['id', 'name', 'updated', 'created', 'open', 'high', 'low', 'volume'], inplace=True)
    df.set_index('dt', inplace=True)
    # Timezone aware date
    df.index = pd.to_datetime(df.index, utc=True)
    # Ensure sorted
    df = df.sort_index()

    # Resample the df according to the interval if not minute by minute
    if training_session.interval != 'M':
        logger.debug(f'Resampling with interval {training_session.interval}')
        df = df['close'].resample(training_session.interval).last().to_frame()

        # Impute missing values
        df = df.fillna(method='ffill')

    # Sanity checks
    logger.debug(f"There are {df['close'].isna().sum()} null values for close")
    assert df['close'].isna().sum() == 0

    return preprocessing(df, training_session, start_date=start_date, end_date=end_date, forecast=forecast)


def denormalize_forecast(forecast, orig_data):
    """
    Convert the predictions back after window normalization
    (only needed if we did windowing)

    Params:
      forecast: our predictions which have been normalized by the first element in
                the orig data window
      orig_data: the original dataset used to make the predictions

    Returns:
         the unnormalized predictions
    """
    new_ps = []
    for n, p in enumerate(forecast):
        w_0 = orig_data[n]
        new_p = (p+1) * w_0
        new_ps.append(new_p)
    return new_ps


def display_results(model, scaler, dataset, dates, window_len, output_size):
    """
    With our predictions we de-normalize the predictions using the inverse transform.

    We plot those prices against the actual prices in the same date range

    We compute the MAE and print it.

    Params:
        model: the training model
        scaler: the scalar we can use to invert the normalization
        dataset: Maybe this is train or test set
        dates: the dates over this set for which we expect predictions
        window_len: how many previous points used when predicting the next
        output_size: how many steps forward are we predicting
    Returns:
        None
    """

    preds = model_forecast(model, dataset, window_len)

    # E.g if window_len is 5, we have predictions for [5:]  since [0, 1, 2, 3, 4] -> [5] etc. If the output_size=1
    # then we neglect the final pred since it uses the final 5 elements of training set to pred a subsequent element, which
    # we have no training data to compare with
    res_df = pd.DataFrame({'y': dataset.flatten()[window_len:], 'yhat': preds.flatten()[:-output_size]})

    # Want to inverse the normalization transform
    if scaler is not None:
        res_df['y_prices'] = scaler.inverse_transform(res_df['y'].values.reshape(-1, 1)).flatten()
        res_df['yhat_prices'] = scaler.inverse_transform(res_df['yhat'].values.reshape(-1, 1)).flatten()
    else:
        # Window scaling
        res_df['y_prices'] = res_df['y']
        res_df['yhat_prices'] = denormalize_forecast(res_df['yhat'], dataset)


    # Plot
    fig = go.Figure()
    fig.add_scatter(x=dates, y=res_df['y_prices'], mode='lines', name="Actual")

    fig.add_scatter(x=dates, y=res_df['yhat_prices'], mode='lines', name="Predicted")

    fig.update_layout(template = 'plotly_dark',
                      xaxis_title="Time",
                      yaxis_title="Price",)


    fig.show()

    # Print the MAE
    # mae = mean_absolute_error(res_df['y'], res_df['yhat'])
    # logger.debug(f'The MAE is {mae}')


def model_forecast(model, series, window_len):
    """
    Take the model we just trained and make predictions.

    We window the dataset then try to predict the next values after the window.
    Note we do not shuffle this time as we are predicting not training, and want to compare also with
    actual prices.

    Parameters:
        model: the ML model trained
        series: the series on which to make predictions
        window_len: size of our window for making preds, e.g. previous 5 elem to predict next perhaps

    Returns:
        the predictions
    """
    # Initially the data is (N,) expand dims to TensorShape([N, 1])
    series = tf.expand_dims(series, axis=-1)

    # Now we just use window_len not +1, because we just want inputs not label, and we predict label
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_len, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_len))

    ds = ds.batch(32).prefetch(1)

    return model.predict(ds)


def denormalize_forecast(forecast, orig_data):
    """
    Convert the predictions back after window normalization
    (only needed if we did windowing)

    Params:
      forecast: our predictions which have been normalized by the first element in
                the orig data window
      orig_data: the original dataset used to make the predictions

    Returns:
         the unnormalized predictions
    """
    new_ps = []
    for n, p in enumerate(forecast):
        w_0 = orig_data[n]
        new_p = (p+1) * w_0
        new_ps.append(new_p)
    return new_ps


def get_forecast_plots(training_session_id, start_date, end_date):

    # Load the Django model corresponding to these options
    try:
        training_session = TrainingSession.objects.get(id=training_session_id)
    except TrainingSession.DoesNotExist as dn_exc:
        logger.error(f'No training session with this id: {training_session_id}')
        return

    qs = training_session.stockprediction_set.all()

    df = read_frame(qs)
    df = df.sort_values(by='dt', ascending=True)

    # Filter date range
    if start_date:
        df = df[df['dt'] >= start_date]
    if end_date:
        df = df[df['dt'] <= end_date]

    graph = []
    for col in ['actual', 'prediction']:
        graph.append(go.Scatter(x=df.dt.tolist(),
                                y=df[col].tolist(),
                                mode='lines',
                                name=col)
                     )

    layout = dict(title=f'Actual vs predicted price over time {training_session.id}',
                  xaxis=dict(title='Date',
                             autotick=True),
                  yaxis=dict(title="Price"),
                  )

    return graph, layout


def get_long_short(predicted_close, current_close):
    """
    Generate the signals long, short

    Parameters
    ----------
    predicted_close : close price predicted at next interval
    current_close: close price now

    Returns
    -------
    long_short : DataFrame
        The long, short, and do nothing signals for each ticker and date
    """
    # If predicted close is greater than the current close then we long
    df1 = (predicted_close >= current_close).astype(int)
    # Otherwise we short if less
    df2 =  -(predicted_close < current_close).astype(int)
    # Combine the 2 (note if no action we'd have False->0 in both)
    return df1 + df2


def get_evaluation_plot(training_session_id, eval_type='train'):

    # Load the Django model corresponding to these options
    try:
        training_session = TrainingSession.objects.get(id=training_session_id)
    except TrainingSession.DoesNotExist as dn_exc:
        logger.error(f'No training session with this id: {training_session_id}')
        return

    qs = training_session.stockprediction_set.all()

    df = read_frame(qs)
    df = df.sort_values(by='dt', ascending=True)

    # Filter date range
    if eval_type == 'train':
        start_date = sorted(training_session.training_dates)[0]
        end_date = sorted(training_session.training_dates)[-1]
    else:
        start_date = sorted(training_session.test_dates)[0]
        end_date = sorted(training_session.test_dates)[-1]
    df = df[df['dt'] >= start_date]
    df = df[df['dt'] <= end_date]

    graph_0 = []
    for col in ['actual', 'prediction']:
        graph_0.append(go.Scatter(x=df.dt.tolist(),
                                y=df[col].tolist(),
                                mode='lines',
                                name=col)
                     )

    layout_0 = dict(title=f'Actual vs predicted price over time ({eval_type} set) {training_session.id}',
                  xaxis=dict(title='Date',
                             autotick=True),
                  yaxis=dict(title="Price"),
                  )

    # Do some profit and significance analysis w/ basic trading strat
    df['actual_shift'] = df['actual'].shift(-1)
    df['prediction_shift'] = df['prediction'].shift(-1)
    df['signal'] = get_long_short(df['prediction_shift'], df['prediction'])
    df['true_signal'] = get_long_short(df['actual_shift'], df['actual'])
    df['return'] = (df['actual_shift'] - df['actual'])/df['actual']
    df['signal_return'] = df['signal'] * df['return']
    df['profit'] = (df['signal_return'].astype(float)+1.0).cumprod()
    df['matches'] = (df['true_signal'] == df['signal']).astype(int)
    mu_null = 0.5
    alpha = 0.05
    pop_mean = df['matches'].mean()
    t, p = stats.ttest_1samp(df['matches'], mu_null)
    # 1 sided
    p = p/2
    logger.debug(f'Got p {p} and t {t}')
    reject_null = False
    if t > 0 and p < alpha:
        # We are looking for greater than 0.5 at level alpha (if t< 0 it means the mean was actually less
        # than 0.5 and no way we reject it the null and claim we are doing better than 50% coin...we may
        # even be doing worse with some signif like 0.4 matches)
        reject_null = True
    graph_1 = [go.Scatter(x=df.dt.tolist(),
                          y=df['profit'].tolist(),
                          mode='lines',
                          name='profit')
               ]

    layout_1 = dict(title=f'Profit over time. Mean: {pop_mean:.2f}, p-value: {p:.2f}, t-score: {t:.2f}.'
                          f'Reject null: {reject_null}',
                    xaxis=dict(title='Date',
                               autotick=True),
                    yaxis=dict(title="% Profit"),
                    )

    return [(graph_0, layout_0), (graph_1, layout_1)]


def get_lending_rates(platform, coin, period, start_date, end_date):

    # Load the Django model corresponding to these options
    lending_rates = LendingRate.objects.filter(platform=platform)
    if coin:
        lending_rates = lending_rates.filter(coin__in=coin)
    if start_date:
        start_date = parse(start_date) if isinstance(start_date, str) else start_date
        lending_rates = lending_rates.filter(dt__gte=start_date)
    if end_date:
        end_date = parse(end_date) if isinstance(end_date, str) else end_date
        lending_rates = lending_rates.filter(dt__lte=end_date)

    df = read_frame(lending_rates)
    df.drop(columns=['id', 'platform', 'previous'], inplace=True)
    # % hourly
    df['hourly_estimate'] = 100*df['estimate']

    # For now just work with annual est
    if period == 'Annual':
        # % annual
        values_name = 'annual_estimate'
        df[values_name] = df['hourly_estimate'].apply(lambda x: 100*(np.power(1+float(x/100), 24*365)-1))
    elif period == 'Daily':
        # % daily
        values_name = 'daily_estimate'
        df[values_name] = df['hourly_estimate'].apply(lambda x: 100*(np.power(1+float(x/100), 24)-1))
    else:
        values_name = 'hourly_estimate'

    pivot_df = df.pivot(index='dt', columns='coin', values=values_name)
    pivot_df = pivot_df.sort_values(by='dt', ascending=True)

    graph = []
    for coin in pivot_df.columns:
        graph.append(go.Scatter(x=pivot_df.index.tolist(),
                                y=pivot_df[coin].tolist(),
                                mode='lines',
                                name=coin)
                     )

    layout = dict(title=f'Lending rate over time for {platform} (% {period} rate)',
                  xaxis=dict(title='Date',
                             autotick=True),
                  yaxis=dict(title="Lending Rate"),
                  )

    return graph, layout