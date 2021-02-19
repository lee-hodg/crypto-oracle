import os
import pickle
import logging
import tensorflow as tf
from django.conf import settings
from django.utils import timezone
from django.core.management.base import BaseCommand
from predictor.models import TrainingSession
from utils.utils import valid_date
from utils.utils import load_data
from predictor.models import Stock


logger = logging.getLogger(__name__)


def build_model(output_size, neurons, activation_func, dropout, loss, optimizer):
    """
    Build the Keras model specified by the params. This will be an LSTM model with an initial 1DConv layer

    :param output_size: e.g. predict 1 point in the future
       neurons: how many neurons for each LSTM later
    :param neurons: How many neurons in each layer
    :param activation_func: the activation function for the hidden layers, e.g. tanh
    :param dropout: regularize with dropout, control to what degree
    :param loss: loss function to use, e.g. mse
    :param optimizer: e.g. adam

    :return: the Keras model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=25, kernel_size=5,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),

        # If drop the Conv1D layer use this as the initial LSTM Layer:
        # tf.keras.layers.LSTM(neurons, input_shape=[None, 1], return_sequences=True, activation=activ_func),

        # Hidden LSTM Layer(s)
        tf.keras.layers.LSTM(neurons, return_sequences=True, activation=activation_func),
        tf.keras.layers.Dropout(float(dropout)),
        #   tf.keras.layers.LSTM(neurons, return_sequences=True, activation=activ_func),
        #   tf.keras.layers.Dropout(dropout),
        #   tf.keras.layers.LSTM(neurons, return_sequences=True, activation=activ_func),
        #   tf.keras.layers.Dropout(dropout),

        # Output LSTM and Dense layer
        tf.keras.layers.LSTM(neurons, return_sequences=False, activation=activation_func),
        tf.keras.layers.Dropout(float(dropout)),
        tf.keras.layers.Dense(units=output_size, activation='linear'),
    ])
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    logger.debug(model.summary())
    return model


class Command(BaseCommand):
    help = '''
        - Load the dataset
        - Optionally load the model weights and fit/refit the model else build the model and fit
        - Make forecasts
        - Save model weights

        run ./manage train_model.py
        '''

    def add_arguments(self, parser):
        parser.add_argument('-N', '--stock-name', dest='name', default='btcusd',
                            type=str, help='Stock name e.g. BTCUSD')
        parser.add_argument('-S', '--start-date', dest='start_date', default='2018-01-01',
                            type=valid_date, help='Start date')
        parser.add_argument('-E', '--end-date', dest='end_date',
                            default=Stock.objects.latest('dt').dt.strftime('%Y-%m-%d %H:%M:%S'),
                            type=valid_date, help='End date')
        parser.add_argument('-X', '--existing-id', dest='existing_id', required=False,
                            type=str, help='ID of already trained model')
        parser.add_argument('-W', '--window-length', dest='window_length', default=15,
                            type=int, help='Window length')
        parser.add_argument('-O', '--output-size', dest='output_size', default=1,
                            type=int, help='Output size to predict')
        parser.add_argument('-F', '--shuffle-size', dest='shuffle_buffer_size', default=1000,
                            type=int, help='Shuffle size for buffer when randomizing pre training')
        parser.add_argument('-T', '--training-split-size', dest='training_size', default=0.8,
                            type=float, help='Train/test set split size')
        parser.add_argument('-U', '--neurons', dest='neurons', default=20,
                            type=int, help='How many neurons in the hidden layer')
        parser.add_argument('-P', '--epochs', dest='epochs', default=4,
                            type=int, help='How many epochs to train over')
        parser.add_argument('-B', '--batch-size', dest='batch_size', default=128,
                            type=int, help='Batch size')
        parser.add_argument('-D', '--dropout', dest='dropout', default=0.25,
                            type=int, help='Drop probability for dropout layers')
        parser.add_argument('-Z', '--optimizer', dest='optimizer', default='adam',
                            type=str, help='Optimizer to use')
        parser.add_argument('-L', '--loss', dest='loss', default='mse',
                            type=str, help='Loss function to use')
        parser.add_argument('-A', '--activation', dest='activation_func', default='tanh',
                            type=str, help='Activation function to use in LSTM hidden layers')
        parser.add_argument('-I', '--interval', dest='interval', default='H',
                            type=str, help='Granularity of data (M, H, D)')
        parser.add_argument('-C', '--scaler', dest='scaler', default='minmax',
                            type=str, help='What scaling to use standard, minmax, robust, window?')

    def handle(self, *args, **options):
        # Get rid of default junk
        unwanted_options = ['verbosity', 'settings', 'pythonpath', 'traceback', 'no_color', 'force_color',
                            'skip_checks']
        [options.pop(el) for el in unwanted_options]

        start_date = options.get('start_date')
        end_date = options.pop('end_date')

        # If we want to update the model to new end date with fresh data
        existing_id = options.pop('existing_id', None)
        if existing_id:
            try:
                training_session = TrainingSession.objects.get(id=existing_id)
            except TrainingSession.DoesNotExist:
                logger.error(f'No training session with id {existing_id}')
        else:
            # Load the Django model corresponding to these options
            training_session, created = TrainingSession.objects.get_or_create(**options)
            if created:
                logger.debug(f'Created a new training training_session with options:\n {options}')
            else:
                logger.debug(f'We already have a training session matching those params.'
                             f'Re-train with -X {training_session.id}')

        # Load the Keras model (note the weights_file path is autogenerated on save)
        if os.path.exists(training_session.weights_path):

            # Maybe we wish to retrain
            last_stock_date = Stock.objects.latest('dt').dt
            if (end_date > training_session.end_date) and (last_stock_date > training_session.end_date):
                # The start date will be from where it ended previously upto now
                start_date = training_session.end_date
                # Max end date is the last stock date we have
                end_date = end_date if end_date < last_stock_date else last_stock_date
                logger.debug(f'Update model weights between date range {start_date} to {end_date}')
            else:
                logger.debug(f'New end date is not greater than current training end date'
                             f' or no need stock data existing after current training end date.')
                return
            model = tf.keras.models.load_model(training_session.weights_path)

        else:
            logger.debug(f'Training for the first time...Build model.')
            tf.keras.backend.clear_session()
            model = build_model(training_session.output_size, training_session.neurons,
                                training_session.activation_func,
                                training_session.dropout, training_session.loss, training_session.optimizer)

        logger.debug(f'Training from {start_date} to {end_date}')
        # Load the data
        sc, windowed_training_data, training_data, test_data, training_dates, test_dates\
            = load_data(training_session, start_date=start_date, end_date=end_date)

        if len(training_data) == 0:
            logger.debug('No new data to update')
            return

        # Fit the model
        training_history = model.fit(windowed_training_data, epochs=training_session.epochs,
                                     batch_size=training_session.batch_size, verbose=1)
        training_session.training_history[timezone.now().isoformat()] = training_history.history

        # Store or update the training/test dates
        training_session.training_dates.extend([d.isoformat() for d in training_dates.to_list()])
        training_session.test_dates.extend([d.isoformat() for d in test_dates.to_list()])

        # Save the model weights
        if os.path.exists(training_session.weights_path):
            # Move old weights file to archive
            old_file = training_session.weights_path
            new_file = os.path.join(settings.BASE_DIR, 'model_weights', 'archive',
                                    f'{str(training_session.id)}_end_date_{training_session.end_date.isoformat()}')
            logger.debug(f'Archive {old_file} to {new_file}')
            os.rename(old_file, new_file)

            # Ensure we update the end date to the new one
            training_session.end_date = end_date

        # Pickle the scaler
        if os.path.exists(training_session.scaler_path):
            old_sc_file = training_session.scaler_path
            new_sc_file = os.path.join(settings.BASE_DIR, 'scalers', 'archive',
                                       f'{str(training_session.id)}_end_date_{training_session.end_date.isoformat()}')
            logger.debug(f'Archive {old_sc_file} to {new_sc_file}')
            os.rename(old_sc_file, new_sc_file)

        # Actually save the keras weights to that dir
        logger.debug(f'Save model weights to {training_session.weights_path}')
        model.save(training_session.weights_path)

        # Pickle the scaler
        logger.debug(f'Pickle the scaler to {training_session.scaler_path}')
        with open(training_session.scaler_path, 'wb') as sc_file:
            pickle.dump(sc, sc_file)

        training_session.save()
