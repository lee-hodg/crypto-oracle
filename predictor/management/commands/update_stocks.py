from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

from predictor.models import Stock
from dateutil.tz import tzutc
from dateutil.parser import parse

from coinapi_rest_v1.restapi import CoinAPIv1


import logging

logger = logging.getLogger(__name__)


def date_utc(s):
    return parse(s, tzinfos={'UTC': tzutc})


class Command(BaseCommand):
    help = '''
        Use the coin API to get latest stocks

        run ./manage update_data.py
        '''

    def add_arguments(self, parser):
        parser.add_argument('-S', '--symbol', dest='symbol', default='BTCUSD',
                            type=str, help='Which stock to scrape?')

    def handle(self, *args, **options):

        symbol = options['symbol']

        if symbol == 'BTCUSD':
            ticker = 'BITSTAMP_SPOT_BTC_USD'
        else:
            return

        # Init the API client
        api = CoinAPIv1(settings.COINAPI_KEY)

        last_datetime = Stock.objects.latest('dt').dt
        current_datetime = timezone.now()
        logger.debug(f'Begin update with current datetime {current_datetime} and last datetime {last_datetime}')

        while last_datetime < current_datetime:
            recent = api.ohlcv_historical_data('BITSTAMP_SPOT_BTC_USD',
                                               {'period_id': '1MIN', 'time_start': last_datetime.isoformat()})

            if not recent:
                logger.debug('No more results')
                break

            logger.debug(f'Got {len(recent)} results to insert.')

            # Bulk insert the new data
            model_instances = [Stock(name=symbol,
                                     dt=res['time_period_end'],
                                     open=res['price_open'],
                                     high=res['price_high'],
                                     low=res['price_low'],
                                     close=res['price_close'],
                                     volume=res['volume_traded']
                                     ) for res in recent]
            Stock.objects.bulk_create(model_instances)

            # Update the latest
            last_datetime = Stock.objects.latest('dt').dt
            logger.debug(f'Updated last datetime to {last_datetime}')
