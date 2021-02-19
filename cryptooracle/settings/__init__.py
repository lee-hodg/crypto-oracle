"""
Settings for different environments are stored in separate files (modules).

.base (base.py) is always loaded first then the setting for the selected environment is
loaded. Environment specific settings can thus reference the values from .base

Environment selection is done in .environment.ENVIRONMENT (environment.py).
"""

#import __builtin__
import sys
import os
from .base import *
from . import environment

level = -1 if sys.version_info < (3, 0) else 1
env_name = os.environ.get('DJANGO_RUNTIME_ENVIRONMENT', environment.ENVIRONMENT)

__imported_module = __import__('%s' % env_name, globals(), locals(), ['*'], level)
locals().update(__imported_module.__dict__)
