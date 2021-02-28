from django.db import models
from model_utils import Choices
from django.utils.translation import ugettext_lazy as _
from django.conf import settings

import os
import uuid

import logging


logger = logging.getLogger(__name__)


class Stock(models.Model):
    """
    Model representing the open, high, low, close, volume of some stock
    (e.g. the price of bitcoin)
    """

    # The name of the stock
    NAMES = Choices(('btcusd', _('BTCUSD')), ('ethusd', _('ETHUSD')))
    name = models.CharField(choices=NAMES, default=NAMES.btcusd, max_length=20, db_index=True)

    # Just keep track of when we created/updated this entry
    updated = models.DateTimeField(auto_now=True, db_index=True)
    created = models.DateTimeField(auto_now_add=True)

    # The datetime of the stock
    dt = models.DateTimeField(null=False, blank=False, db_index=True)

    # The price data
    open = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)
    high = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)
    low = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)
    close = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)
    volume = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)

    def __repr__(self):
        return f'{self.name}: {self.dt}'

    class Meta:
        unique_together = ('name', 'dt')


class TrainingSession(models.Model):
    """
    Represents a trained model. We can use this to organize better trained models with
    different parameters. Load them, update their training, make predictions from them etc.
    """
    # This will be where save the model weights
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Evaluation sessions will have the train/test split and won't be updated with bleeding edge stocks
    # production models will use all available data. Training until the bleeding edge
    evaluation_session = models.BooleanField(default=True)

    # The name of the stock
    NAMES = Choices(('btcusd', _('BTCUSD')), ('ethusd', _('ETHUSD')))
    name = models.CharField(choices=NAMES, default=NAMES.btcusd, max_length=20)

    # The training data range
    start_date = models.DateTimeField(null=False, blank=False)
    end_date = models.DateTimeField(null=False, blank=False)

    # How far back in the sequence to look when predicting the next values
    window_length = models.PositiveSmallIntegerField(null=False, blank=False, default=15)
    output_size = models.PositiveSmallIntegerField(null=False, blank=False, default=1)

    # How many nodes in the LSTM hidden layers
    neurons = models.PositiveSmallIntegerField(null=False, blank=False, default=20)

    # Size of the shuffle buffer (used when randomizing training data)
    shuffle_buffer_size = models.PositiveSmallIntegerField(null=False, blank=False, default=1000)

    # train/test split size
    training_size = models.DecimalField(max_digits=3, decimal_places=2, null=False, blank=False, default=0.8)

    # How many epochs were training over
    epochs = models.PositiveSmallIntegerField(null=True, blank=True, default=4)

    # Size of batches
    batch_size = models.PositiveSmallIntegerField(null=True, blank=True, default=128)

    # Drop probability for dropout regularization layers
    dropout = models.DecimalField(max_digits=3, decimal_places=2, null=False, blank=False, default=0.25)

    # The name of the optimizer
    OPT_NAMES = Choices(('adam', _('ADAM')), )
    optimizer = models.CharField(choices=OPT_NAMES, default=OPT_NAMES.adam, max_length=20)

    # The name of the loss function
    LOSS_NAMES = Choices(('mse', _('MSE')), ('mae', _('MAE')) )
    loss = models.CharField(choices=LOSS_NAMES, default=LOSS_NAMES.mse, max_length=20)

    # The name of the hidden layer activation function
    ACT_NAMES = Choices(('tanh', _('Tanh')), )
    activation_func = models.CharField(choices=ACT_NAMES, default=ACT_NAMES.tanh, max_length=20)

    # We could look at minutes, hours, days
    INTERVAL = Choices(('M', _('MINUTE')), ('H', _('HOUR')), ('D', _('DAILY')))
    interval = models.CharField(choices=INTERVAL, default=INTERVAL.M, max_length=20)

    # Store the history, such as loss and mae evolution
    training_history = models.JSONField(null=True, blank=True, default=dict)

    # The training and test set dates
    training_dates = models.JSONField(null=True, blank=True, default=list)
    test_dates = models.JSONField(null=True, blank=True, default=list)

    SCALERS = Choices(('window', _('WINDOW')), ('standard', _('STANDARD')),
                      ('minmax', _('MINMAX')), ('robust', _('ROBUST'))
                      )
    scaler = models.CharField(choices=SCALERS, default=SCALERS.minmax, max_length=20)

    def get_params(self):
        return '__'.join([f'name_{self.name}',
                          f'start_date_{self.start_date.isoformat()}'
                          f'end_date_{self.end_date.isoformat()}',
                          f'window_len_{self.window_length}',
                          f'output_size_{self.output_size}',
                          f'shuffle_{self.shuffle_buffer_size}',
                          f'training_size_{self.training_size}',
                          f'neurons_{self.neurons}',
                          f'epochs_{self.epochs}',
                          f'batch_{self.batch_size}',
                          f'dropout_{self.dropout}',
                          f'opt_{self.optimizer}',
                          f'loss_{self.loss}',
                          f'act_{self.activation_func}',
                          f'int_{self.interval}',
                          f'sc_{self.scaler}'
                          ])

    @property
    def weights_path(self):
        return os.path.join(settings.BASE_DIR,  'model_weights', str(self.id))

    @property
    def scaler_path(self):
        """
        Where did we pickle the scaler used during training
        :return:
        """
        return os.path.join(settings.BASE_DIR,  'scalers', f'{str(self.id)}_scaler.p')

    def __repr__(self):
        return self.get_params()

    class Meta:
        unique_together = ('name', 'start_date', 'end_date', 'window_length', 'output_size',
                           'evaluation_session',
                           'neurons', 'shuffle_buffer_size', 'training_size', 'epochs', 'batch_size',
                           'dropout', 'optimizer', 'loss', 'activation_func')


class StockPrediction(models.Model):
    """
    Model representing the close price prediction for a given date by a given ML model
    """

    training_session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)

    # The datetime of the stock prediction
    dt = models.DateTimeField(null=False, blank=False, db_index=True)

    # The price data
    prediction = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)
    actual = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)

    def __repr__(self):
        return f'{self.dt}: {self.prediction}'

    class Meta:
        unique_together = ('training_session', 'dt')


class LendingRate(models.Model):
    """
    Model representing lending rate from ftx or other platforms
    """
    # The name of the stock
    PLATFORMS = Choices(('ftx', _('FTX')), ('kucoin', _('KuCoin')))
    platform = models.CharField(choices=PLATFORMS, default=PLATFORMS.ftx, max_length=20)

    # Which coin/fiat are we lending?
    coin = models.CharField(max_length=20)

    # The datetime the rate was obtained
    dt = models.DateTimeField(null=False, blank=False, db_index=True)

    # The hourly lending rate data
    previous = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)
    estimate = models.DecimalField(max_digits=19, decimal_places=10, null=True, blank=False)

    # term of loan
    term = models.PositiveSmallIntegerField(null=True)

    @property
    def day_estimate(self):
        return ((1+self.estimate)**24)-1

    @property
    def annual_estimate(self):
        return ((1+self.estimate)**(24*365))-1

    def __repr__(self):
        return f'{self.dt}: {self.platform}: {self.coin}: {self.estimate}'

    class Meta:
        unique_together = ('platform', 'coin', 'dt', 'term')
        ordering = ['dt']
