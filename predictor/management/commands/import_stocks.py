from django.core.management.base import BaseCommand
from django.conf import settings
from predictor.models import Stock
from utils.utils import valid_date
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = '''
        Import data from csv to the database.
        Initial import.

        run ./manage import_data.py
        '''

    def add_arguments(self, parser):
        parser.add_argument('-D', '--dir', dest='directory', default='data/btc',
                            type=str, help='Directory to read data from')

    def handle(self, *args, **options):
        directory = options['directory']
        stock_name = directory.split('/')[-1]
        directory = os.path.join(settings.BASE_DIR, directory)
        logger.debug(f'Importing data from {directory}')

        # Walk dir to read in all csv files
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    with open(os.path.join(root, file), 'r') as csv_file:
                        logger.debug(f'Reading CSV from {csv_file}...')
                        df = pd.read_csv(csv_file, low_memory=False, skiprows=1)
                                         # parse_dates=[1], date_parser=date_utc)
                        df.drop(columns='Unix Timestamp', inplace=True)
                        df.columns = df.columns.str.lower()
                        logger.debug(f'Dataframe with {df.shape[0]} rows.')

                        # Bulk insert the new data
                        model_instances = [Stock(name=stock.symbol,
                                                 dt=valid_date(stock.date),
                                                 open=float(stock.open) if stock.open else None,
                                                 high=float(stock.high) if stock.high else None,
                                                 low=float(stock.low) if stock.low else None,
                                                 close=float(stock.close) if stock.close else None,
                                                 volume=float(stock.volume) if stock.volume else None
                                                 ) for stock in df.itertuples()]
                        Stock.objects.bulk_create(model_instances)
