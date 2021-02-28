from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

from predictor.models import LendingRate

from kucoin import client as kucoin_client
import ftx
import numpy as np

import logging

logger = logging.getLogger(__name__)


def daily_to_hourly(daily_rate):
    """
    Convert the daily rate to hourly
    :param daily_rate:
    :return:
    """
    return np.power(daily_rate+1, 1/24)-1


class Command(BaseCommand):
    help = '''
        Refresh out lending rates
        
        run ./manage update_lending_rate.py
        '''

    def add_arguments(self, parser):
        parser.add_argument('-P', '--platform', dest='platform', default='ftx',
                            type=str, help='Which platform to pull?')

    def handle(self, *args, **options):

        platform = options['platform']
        logger.debug(f'Pulling data for {platform}')

        if platform not in ['kucoin', 'ftx']:
            logger.debug('Currently only ftx and kucoin are supported.')
            return

        current_datetime = timezone.now()

        if platform == 'ftx':
            # Init the API client
            client = ftx.FtxClient(api_key=settings.FTX_API_KEY, api_secret=settings.FTX_API_SECRET)

            lending_rates = client._get('spot_margin/lending_rates')

            logger.debug(f'Insert lending rates: {lending_rates}')

            # Bulk insert the new data
            model_instances = [LendingRate(platform=platform,
                                           coin=res['coin'],
                                           dt=current_datetime,
                                           previous=res['previous'],
                                           estimate=res['estimate'])
                               for res in lending_rates]
            LendingRate.objects.bulk_create(model_instances)
        elif platform == 'kucoin':
            # Init the Kucoin client
            market_client = kucoin_client.MarginData()
            currencies = ['USDT', 'USDC', 'BTC', 'ETH']
            terms = [7, 14, 28]
            lending_rates = []
            for currency in currencies:
                for term in terms:
                    # lending_market = market_client.get_lending_market(currency, term=term)
                    lending_market = market_client.get_margin_data(currency)
                    lending_market = [f for f in lending_market if f['term'] == term]
                    count_els = len(lending_market)
                    if count_els == 0:
                        logger.debug(f'No offers for {currency} ({term} days).Skip..')
                        continue
                    total_size = sum([float(l['size']) for l in lending_market])
                    mean_rate = sum([float(l['dailyIntRate']) for l in lending_market])/count_els
                    mean_hourly_rate = daily_to_hourly(mean_rate)
                    logger.debug(f'For {currency} ({term} days) mean rate is {mean_rate}')
                    weighted_rate = sum([float(l['dailyIntRate'])*float(l['size']) for l in lending_market])/total_size
                    weighted_hourly_rate = daily_to_hourly(weighted_rate)

                    lending_rate = {'coin': currency,
                                    'dt': current_datetime,
                                    'previous': weighted_hourly_rate,
                                    'estimate': weighted_hourly_rate,
                                    'term': term
                                    }
                    lending_rates.append(lending_rate)
            # Bulk insert the new data
            model_instances = [LendingRate(platform=platform,
                                           coin=res['coin'],
                                           dt=current_datetime,
                                           previous=res['previous'],
                                           estimate=res['estimate'],
                                           term=res['term'])
                               for res in lending_rates]
            LendingRate.objects.bulk_create(model_instances)
