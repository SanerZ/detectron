# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:06:03 2018

@author: ADMIN
"""

import logging
import logging.config
from functools import partial
from pathlib2 import Path

cfgpath = Path(__file__).parent/'logging.conf'

lcfg = partial(logging.config.fileConfig, cfgpath.as_posix())

pl = logging.getLogger('file_append')
ps = logging.getLogger('file_only')