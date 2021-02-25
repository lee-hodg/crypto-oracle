from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

from predictor.models import LendingRate

import ftx

import logging

logger = logging.getLogger(__name__)


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

        if platform != 'ftx':
            logger.debug('Currently only ftx is supported.')
            return

        # Init the API client
        client = ftx.FtxClient(api_key=settings.FTX_API_KEY, api_secret=settings.FTX_API_SECRET)

        current_datetime = timezone.now()

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

