"""
Development settings
"""
from .base import *

DEBUG = True


try:
    from .local import *
except ImportError:
    # local settings is not mandatory
    pass


INSTALLED_APPS += (
    'django_extensions',
)

# Django debug toolbar things

MIDDLEWARE = ['debug_toolbar.middleware.DebugToolbarMiddleware'] + MIDDLEWARE

# Want this early if we are going to use it, but after static files
stat_files_idx = INSTALLED_APPS.index('django.contrib.staticfiles')
if DEBUG:
    INSTALLED_APPS.insert(stat_files_idx+1, 'debug_toolbar')

# Debug toolbar
INTERNAL_IPS = [
    '127.0.0.1',
]


ALLOWED_HOSTS = ['*']

# log_conf = {'handlers': ['console'],
#             'level': 'DEBUG'}
# LOGGING['loggers'].update({'artwork': log_conf, 'user': log_conf, 'sys_util': log_conf, 'devices': log_conf})
LOGGING['formatters']['verbose']['()'] = 'coloredlogs.ColoredFormatter'

# db_log_conf = {'handlers': ['console'],
#                'filters': ['slow_queries'],
#                'level': 'DEBUG'}
# LOGGING['loggers'].update({'django.db.backends': db_log_conf})

